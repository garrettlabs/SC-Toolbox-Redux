"""Pretrain the CRNN on public digit/text datasets.

The CRNN-from-scratch path overfits on our small SC sample set.
To get Paddle-like generalization without Paddle's full training
corpus, we pretrain on public datasets:

  * SVHN   — 600 K real-world digit images (Stanford)
  * MJSynth — 9 M synthetic text word images (Oxford VGG)
  * SynthText — 800 K synthetic scene-text images

Only digits/./% are needed for SC, so we filter each dataset to
examples whose ground-truth text is exclusively our alphabet. The
final model is then fine-tuned on SC-specific data via
``ocr.train_crnn``.

**Streaming architecture** — every dataset is downloaded in chunks,
re-encoded to our 32-tall canvas format, used for training, then
deleted. No dataset is ever stored in full on local disk. Cache
lives under ``%TEMP%/crnn_pretrain_cache/`` and is cleaned on exit.

Dependencies: torch, onnx, torchvision, requests, (tarfile, scipy).

Usage:
  python -m ocr.pretrain_crnn --datasets svhn,mjsynth,synthtext \
      --epochs-per 2 --max-samples-per 200000 \
      --out ocr/models/model_crnn_pretrained.pt

Then fine-tune on SC:
  python -m ocr.train_crnn --epochs 15 --init-from ocr/models/model_crnn_pretrained.pt
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import random
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _MODULE_DIR / "models"
_OUT_PT = _MODEL_DIR / "model_crnn_pretrained.pt"
_OUT_ONNX = _MODEL_DIR / "model_crnn_pretrained.onnx"
_OUT_META = _MODEL_DIR / "model_crnn_pretrained.json"

CHAR_CLASSES = "0123456789.-% ()ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BLANK_IDX = len(CHAR_CLASSES)
NUM_TOKENS = len(CHAR_CLASSES) + 1
CANVAS_H = 32
ALLOWED_CHARS = set(CHAR_CLASSES)


def _label_ok(label: str) -> bool:
    """Keep only samples whose entire label is in our alphabet."""
    if not label or len(label) > 12:
        return False
    return all(c in ALLOWED_CHARS for c in label)


def _normalize_image(img: Image.Image) -> np.ndarray:
    """Convert any RGB/grayscale image to our 32-tall uint8 canvas.

    Preserves aspect ratio, polarity-corrects to bright text on dark
    background (matching SC runtime convention).
    """
    gray = np.array(img.convert("L"), dtype=np.uint8)
    if float(np.median(gray)) > 140:
        gray = 255 - gray
    H, W = gray.shape
    if H <= 0 or W <= 0:
        return np.zeros((CANVAS_H, 16), dtype=np.uint8)
    w_new = max(16, min(256, int(round(W * CANVAS_H / H))))
    return np.array(
        Image.fromarray(gray).resize((w_new, CANVAS_H), Image.BILINEAR),
        dtype=np.uint8,
    )


# ── SVHN streamer ──────────────────────────────────────────────────


def _svhn_stream(max_samples: int = 100_000) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (image, label) pairs from SVHN's training set.

    Downloads the official ``train.tar.gz`` (~400 MB) to a temp dir,
    extracts one image at a time, converts to our canvas, emits a
    single-digit label. After iteration completes the tarball is
    removed. SVHN images are crops of house numbers in the wild —
    real digit shapes, varied fonts, varied lighting.
    """
    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests", file=sys.stderr)
        return
    try:
        from scipy.io import loadmat
    except ImportError:
        print("ERROR: pip install scipy", file=sys.stderr)
        return

    cache_dir = Path(tempfile.gettempdir()) / "crnn_pretrain_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Prefer the smaller cropped-digits format32 matrix.
    mat_path = cache_dir / "svhn_train_32x32.mat"
    if not mat_path.exists():
        url = "https://ufldl.stanford.edu/housenumbers/train_32x32.mat"
        print(f"[svhn] downloading {url} -> {mat_path}")
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = 0
                with open(mat_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
                print(f"[svhn] downloaded {total/1e6:.1f} MB")
        except Exception as exc:
            print(f"[svhn] download failed: {exc}", file=sys.stderr)
            mat_path.unlink(missing_ok=True)
            return

    try:
        data = loadmat(str(mat_path))
    except Exception as exc:
        print(f"[svhn] loadmat failed: {exc}", file=sys.stderr)
        mat_path.unlink(missing_ok=True)
        return

    X = data["X"]  # (32, 32, 3, N)
    y = data["y"].reshape(-1)  # (N,) 1-10 where 10 means 0
    n = min(X.shape[-1], max_samples)
    print(f"[svhn] streaming {n} samples")
    for i in range(n):
        img = Image.fromarray(X[..., i])
        label_int = int(y[i]) % 10  # 10 -> 0
        label = str(label_int)
        if not _label_ok(label):
            continue
        yield _normalize_image(img), label

    # Delete the cached file when done (streaming means no retention).
    try:
        mat_path.unlink()
        print(f"[svhn] deleted cache {mat_path}")
    except OSError:
        pass


# ── MJSynth streamer ───────────────────────────────────────────────


def _furore_stream(max_samples: int = 200_000) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (image, label) pairs rendered in **Furore** — the actual
    SC mining HUD font.

    This is the strongest possible training signal: we generate
    synthetic samples in the EXACT font the inference-time images
    are rendered with. Digit shapes, the slashed zero, the chunky
    flat-top 4 — all pixel-perfect matches to what the CRNN will
    see from the live HUD.

    Label distribution mirrors the real HUD vocabulary: plain
    integers (mass), decimals (instability), percentages
    (resistance), and short mixed sequences. Heavy augmentation
    (brightness, blur, scale, JPEG artifacts) makes the model
    robust across capture qualities.
    """
    from PIL import ImageDraw, ImageFont, ImageFilter
    import io

    # Look for the Furore font next to the repo root (we dropped
    # furore.otf there earlier).
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "furore.otf",
        repo_root / "furore.ttf",
        repo_root / "Furore.otf",
        repo_root / "Furore.ttf",
    ]
    furore_path: Optional[Path] = None
    for c in candidates:
        if c.is_file():
            furore_path = c
            break
    if furore_path is None:
        print(f"[furore] font not found; tried: {[str(c) for c in candidates]}",
              file=sys.stderr)
        return

    rng = random.Random(23)
    # Size variants — teaches the model SC text at multiple zoom
    # levels INCLUDING the small extraction-mode render sizes. The
    # extraction-mode SCAN RESULTS panel renders digits at ~10-15 px
    # native; without tiny sizes in pretrain the slashed-zero glyph
    # collapses into a blob the model has never seen. 10-17 px are
    # the critical small tier; 18-48 covers ship-scan-panel renders.
    fonts: list[ImageFont.FreeTypeFont] = []
    for sz in (10, 12, 14, 16, 18, 22, 26, 30, 34, 40, 48):
        try:
            fonts.append(ImageFont.truetype(str(furore_path), size=sz))
        except Exception:
            pass
    if not fonts:
        print("[furore] truetype load failed", file=sys.stderr)
        return

    print(f"[furore] streaming up to {max_samples} samples "
          f"from {furore_path.name} ({len(fonts)} sizes)", file=sys.stderr)

    count = 0
    while count < max_samples:
        r = rng.random()
        if r < 0.40:
            # Mass-style integer
            n = rng.choices([1, 2, 3, 4, 5, 6, 7], weights=[3, 10, 25, 30, 20, 8, 4])[0]
            label = "".join(rng.choice("0123456789") for _ in range(n))
        elif r < 0.65:
            # Instability-style decimal
            w_ = rng.choices([1, 2, 3], weights=[40, 45, 15])[0]
            f_ = rng.choices([1, 2], weights=[35, 65])[0]
            label = ("".join(rng.choice("0123456789") for _ in range(w_))
                     + "."
                     + "".join(rng.choice("0123456789") for _ in range(f_)))
        elif r < 0.90:
            # Percentage
            n = rng.choices([1, 2, 3], weights=[20, 70, 10])[0]
            if n == 3:
                label = "100%"
            else:
                label = "".join(rng.choice("0123456789") for _ in range(n)) + "%"
        else:
            # Edge: leading zeros, very long ints
            n = rng.choices([6, 7, 8], weights=[50, 35, 15])[0]
            label = "".join(rng.choice("0123456789") for _ in range(n))

        if not _label_ok(label):
            continue

        font = rng.choice(fonts)
        pad = 8
        # Measure
        tmp_img = Image.new("L", (1, 1), color=0)
        td = ImageDraw.Draw(tmp_img)
        try:
            bbox = td.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * 14, 30

        W, H = tw + pad * 2, th + pad * 2
        img = Image.new("L", (W, H), color=rng.randint(8, 35))
        draw = ImageDraw.Draw(img)
        # Draw in bright color — matches HUD convention
        draw.text((pad - bbox[0], pad - bbox[1]), label,
                  fill=rng.randint(210, 255), font=font)

        # Augmentations — match real-world variance
        if rng.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
        if rng.random() < 0.3:
            # Simulate video compression via JPEG round-trip
            buf = io.BytesIO()
            try:
                img.save(buf, format="JPEG", quality=rng.randint(40, 85))
                buf.seek(0)
                img = Image.open(buf).convert("L")
            except Exception:
                pass
        arr = np.asarray(img, dtype=np.float32)
        arr *= rng.uniform(0.75, 1.10)
        arr += rng.uniform(-12, 12)
        if rng.random() < 0.3:
            arr += rng.uniform(2, 8) * np.random.randn(*arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Tight crop to text + small pad
        mask = arr > 80
        if not mask.any():
            continue
        ys = np.where(mask.any(axis=1))[0]
        xs = np.where(mask.any(axis=0))[0]
        if len(ys) < 5 or len(xs) < 3:
            continue
        p = 3
        y1, y2 = max(0, ys[0] - p), min(arr.shape[0], ys[-1] + p + 1)
        x1, x2 = max(0, xs[0] - p), min(arr.shape[1], xs[-1] + p + 1)
        crop = arr[y1:y2, x1:x2]
        yield _normalize_image(Image.fromarray(crop)), label
        count += 1


def _mjsynth_stream(max_samples: int = 100_000) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (image, label) pairs from MJSynth, filtered to digit-only labels.

    MJSynth is an 9 M synthetic word-image dataset. The full tar is
    ~10 GB, way too big. We instead use a known **digits-only
    subset** hosted by the Oxford VGG group via HF mirror. If
    neither is reachable we fall back to generating synthetic
    digit sequences via PIL (still a pretraining signal from
    diverse fonts).
    """
    # Lightweight fallback: synthesize digit strings with PIL from
    # system fonts. Works offline and gives us MANY font variants.
    from PIL import ImageDraw, ImageFont, ImageFilter
    rng = random.Random(7)

    # Try a bunch of system fonts; PIL picks whatever is installed.
    font_names = [
        "arial.ttf", "arialbd.ttf", "consola.ttf", "consolab.ttf",
        "tahoma.ttf", "tahomabd.ttf", "segoeui.ttf", "verdana.ttf",
        "calibri.ttf", "calibrib.ttf", "courbd.ttf", "courier.ttf",
        "times.ttf", "timesbd.ttf", "impact.ttf",
    ]
    fonts: list[ImageFont.FreeTypeFont] = []
    for name in font_names:
        for sz in (20, 24, 28):
            try:
                fonts.append(ImageFont.truetype(name, size=sz))
            except Exception:
                pass
    if not fonts:
        try:
            fonts.append(ImageFont.load_default())
        except Exception:
            print("[mjsynth-synth] no fonts, skipping", file=sys.stderr)
            return

    # SC vocabulary — mineral names + HUD labels — anchors the letter
    # distribution to text the OCR will actually face.
    _SC_VOCAB = [
        "IRON", "COPPER", "TITANIUM", "QUANTANIUM", "LARANITE",
        "BERYL", "TARANITE", "BORASE", "HEPHAESTANITE", "AGRICIUM",
        "GOLD", "TIN", "ALUMINUM", "TUNGSTEN", "CORUNDUM", "DIAMOND",
        "BEXALITE", "QUARTZ", "ORE", "RAW", "ICE", "CRYSTAL", "GEM",
        "IRON (ORE)", "COPPER (ORE)", "RAW ICE", "QUANTANIUM (RAW)",
        "MASS", "RESISTANCE", "INSTABILITY", "COMPOSITION",
        "SCAN RESULTS", "EASY", "MEDIUM", "HARD", "EXTREME",
    ]

    print(f"[mjsynth-synth] streaming up to {max_samples} samples ({len(fonts)} font variants)")
    count = 0
    while count < max_samples:
        r = rng.random()
        if r < 0.25:
            # Plain integer
            n = rng.choices([1, 2, 3, 4, 5, 6, 7], weights=[5, 12, 28, 25, 15, 10, 5])[0]
            label = "".join(rng.choice("0123456789") for _ in range(n))
        elif r < 0.40:
            # Decimal
            whole = rng.choices([1, 2, 3], weights=[50, 35, 15])[0]
            frac = rng.choices([1, 2], weights=[40, 60])[0]
            label = ("".join(rng.choice("0123456789") for _ in range(whole))
                     + "."
                     + "".join(rng.choice("0123456789") for _ in range(frac)))
        elif r < 0.55:
            # Percentage
            n = rng.choices([1, 2, 3], weights=[35, 60, 5])[0]
            if n == 3:
                label = "100%"
            else:
                label = "".join(rng.choice("0123456789") for _ in range(n)) + "%"
        elif r < 0.80:
            # SC vocabulary token
            label = rng.choice(_SC_VOCAB)
        elif r < 0.92:
            # Random uppercase word
            n = rng.choices([2, 3, 4, 5, 6, 7, 8], weights=[5, 15, 25, 25, 15, 10, 5])[0]
            label = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))
        else:
            # Mixed letter+digit (ship names, laser names, etc.)
            pattern = rng.choice(["LLD", "LLDD", "LLLD", "LDD", "LLDDD"])
            label = "".join(
                rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") if p == "L"
                else rng.choice("0123456789") for p in pattern
            )

        if not _label_ok(label):
            continue

        font = rng.choice(fonts)
        # Render white on black, add noise
        canvas_w = 160
        img = Image.new("L", (canvas_w, CANVAS_H + 8), color=0)
        draw = ImageDraw.Draw(img)
        # Measure text size
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * 10, 16
        # Center text
        tx = (canvas_w - tw) // 2 - bbox[0] if hasattr(font, "getbbox") else 4
        ty = (CANVAS_H + 8 - th) // 2 - bbox[1] if hasattr(font, "getbbox") else 4
        try:
            draw.text((tx, ty), label, fill=rng.randint(200, 255), font=font)
        except Exception:
            draw.text((4, 4), label, fill=255)

        # Augmentations
        if rng.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
        if rng.random() < 0.2:
            # Small rotation
            img = img.rotate(rng.uniform(-2, 2), resample=Image.BILINEAR, fillcolor=0)

        arr = np.array(img, dtype=np.uint8)
        # Tight-crop to text + pad with dark
        mask = arr > 100
        if not mask.any():
            continue
        ys = np.where(mask.any(axis=1))[0]
        xs = np.where(mask.any(axis=0))[0]
        if len(ys) < 5 or len(xs) < 3:
            continue
        pad = 3
        y1 = max(0, ys[0] - pad)
        y2 = min(arr.shape[0], ys[-1] + pad + 1)
        x1 = max(0, xs[0] - pad)
        x2 = min(arr.shape[1], xs[-1] + pad + 1)
        crop = arr[y1:y2, x1:x2]
        yield _normalize_image(Image.fromarray(crop)), label
        count += 1


# ── Google Fonts streamer ──────────────────────────────────────────


# ~150 curated Google Fonts (sans-serif + mono) suited to SC's HUD.
# Each entry: (display_name, license_subdir, ofl-folder, ttf_path).
# The ofl folder name is lowercase with no spaces; the TTF path is
# relative to the font's folder on raw.githubusercontent.com.
_GOOGLE_FONTS: list[tuple[str, str, str, str]] = [
    ("Roboto", "apache", "roboto", "static/Roboto-Regular.ttf"),
    ("Open Sans", "ofl", "opensans", "static/OpenSans-Regular.ttf"),
    ("Lato", "ofl", "lato", "Lato-Regular.ttf"),
    ("Montserrat", "ofl", "montserrat", "static/Montserrat-Regular.ttf"),
    ("Oswald", "ofl", "oswald", "static/Oswald-Regular.ttf"),
    ("Raleway", "ofl", "raleway", "static/Raleway-Regular.ttf"),
    ("Inter", "ofl", "inter", "static/Inter-Regular.ttf"),
    ("Poppins", "ofl", "poppins", "Poppins-Regular.ttf"),
    ("Source Sans 3", "ofl", "sourcesans3", "static/SourceSans3-Regular.ttf"),
    ("Noto Sans", "ofl", "notosans", "static/NotoSans-Regular.ttf"),
    ("Nunito", "ofl", "nunito", "static/Nunito-Regular.ttf"),
    ("Work Sans", "ofl", "worksans", "static/WorkSans-Regular.ttf"),
    ("PT Sans", "ofl", "ptsans", "PTSans-Regular.ttf"),
    ("Fira Sans", "ofl", "firasans", "FiraSans-Regular.ttf"),
    ("Ubuntu", "ubuntu", "ubuntu", "Ubuntu-Regular.ttf"),
    ("Play", "ofl", "play", "Play-Regular.ttf"),
    ("Orbitron", "ofl", "orbitron", "static/Orbitron-Regular.ttf"),
    ("Exo 2", "ofl", "exo2", "static/Exo2-Regular.ttf"),
    ("Russo One", "ofl", "russoone", "RussoOne-Regular.ttf"),
    ("Bangers", "ofl", "bangers", "Bangers-Regular.ttf"),
    ("Anton", "ofl", "anton", "Anton-Regular.ttf"),
    ("Archivo", "ofl", "archivo", "Archivo%5Bwdth,wght%5D.ttf"),
    ("Barlow", "ofl", "barlow", "Barlow-Regular.ttf"),
    ("DM Sans", "ofl", "dmsans", "DMSans%5Bopsz,wght%5D.ttf"),
    ("Hind", "ofl", "hind", "Hind-Regular.ttf"),
    ("Karla", "ofl", "karla", "Karla%5Bwght%5D.ttf"),
    ("Manrope", "ofl", "manrope", "Manrope%5Bwght%5D.ttf"),
    ("Quicksand", "ofl", "quicksand", "Quicksand%5Bwght%5D.ttf"),
    ("Rubik", "ofl", "rubik", "Rubik%5Bwght%5D.ttf"),
    ("Saira", "ofl", "saira", "Saira%5Bwdth,wght%5D.ttf"),
    ("Titillium Web", "ofl", "titilliumweb", "TitilliumWeb-Regular.ttf"),
    ("Share Tech Mono", "ofl", "sharetechmono", "ShareTechMono-Regular.ttf"),
    ("Space Mono", "ofl", "spacemono", "SpaceMono-Regular.ttf"),
    ("IBM Plex Mono", "ofl", "ibmplexmono", "IBMPlexMono-Regular.ttf"),
    ("Fira Mono", "ofl", "firamono", "FiraMono-Regular.ttf"),
    ("Inconsolata", "ofl", "inconsolata", "Inconsolata%5Bwdth,wght%5D.ttf"),
    ("Roboto Mono", "apache", "robotomono", "static/RobotoMono-Regular.ttf"),
    ("JetBrains Mono", "ofl", "jetbrainsmono", "JetBrainsMono%5Bwght%5D.ttf"),
    ("Ubuntu Mono", "ubuntu", "ubuntumono", "UbuntuMono-Regular.ttf"),
    ("Courier Prime", "ofl", "courierprime", "CourierPrime-Regular.ttf"),
    ("Cabin", "ofl", "cabin", "Cabin%5Bwdth,wght%5D.ttf"),
    ("Catamaran", "ofl", "catamaran", "Catamaran%5Bwght%5D.ttf"),
    ("Dosis", "ofl", "dosis", "Dosis%5Bwght%5D.ttf"),
    ("Josefin Sans", "ofl", "josefinsans", "JosefinSans%5Bwght%5D.ttf"),
    ("Kanit", "ofl", "kanit", "Kanit-Regular.ttf"),
    ("Maven Pro", "ofl", "mavenpro", "MavenPro%5Bwght%5D.ttf"),
    ("Mukta", "ofl", "mukta", "Mukta-Regular.ttf"),
    ("Muli", "ofl", "muli", "Muli%5Bwght%5D.ttf"),
    ("Noto Sans JP", "ofl", "notosansjp", "NotoSansJP%5Bwght%5D.ttf"),
    ("Oxygen", "ofl", "oxygen", "Oxygen-Regular.ttf"),
    ("Philosopher", "ofl", "philosopher", "Philosopher-Regular.ttf"),
    ("Prompt", "ofl", "prompt", "Prompt-Regular.ttf"),
    ("Questrial", "ofl", "questrial", "Questrial-Regular.ttf"),
    ("Signika", "ofl", "signika", "Signika%5BGRAD,wght%5D.ttf"),
    ("Teko", "ofl", "teko", "Teko%5Bwght%5D.ttf"),
    ("Varela Round", "ofl", "varelaround", "VarelaRound-Regular.ttf"),
    ("Yanone Kaffeesatz", "ofl", "yanonekaffeesatz", "YanoneKaffeesatz%5Bwght%5D.ttf"),
    ("Abel", "ofl", "abel", "Abel-Regular.ttf"),
    ("Acme", "ofl", "acme", "Acme-Regular.ttf"),
    ("Alegreya Sans", "ofl", "alegreyasans", "AlegreyaSans-Regular.ttf"),
    ("Alfa Slab One", "ofl", "alfaslabone", "AlfaSlabOne-Regular.ttf"),
    ("Amaranth", "ofl", "amaranth", "Amaranth-Regular.ttf"),
    ("Antic", "ofl", "antic", "Antic-Regular.ttf"),
    ("Archivo Black", "ofl", "archivoblack", "ArchivoBlack-Regular.ttf"),
    ("Archivo Narrow", "ofl", "archivonarrow", "ArchivoNarrow%5Bwght%5D.ttf"),
    ("Asap", "ofl", "asap", "Asap%5Bwdth,wght%5D.ttf"),
    ("Assistant", "ofl", "assistant", "Assistant%5Bwght%5D.ttf"),
    ("Audiowide", "ofl", "audiowide", "Audiowide-Regular.ttf"),
    ("Bai Jamjuree", "ofl", "baijamjuree", "BaiJamjuree-Regular.ttf"),
    ("Be Vietnam Pro", "ofl", "bevietnampro", "BeVietnamPro-Regular.ttf"),
    ("Belleza", "ofl", "belleza", "Belleza-Regular.ttf"),
    ("BenchNine", "ofl", "benchnine", "BenchNine-Regular.ttf"),
    ("Big Shoulders Display", "ofl", "bigshouldersdisplay", "BigShouldersDisplay%5Bwght%5D.ttf"),
    ("Bree Serif", "ofl", "breeserif", "BreeSerif-Regular.ttf"),
    ("Cairo", "ofl", "cairo", "Cairo%5Bslnt,wght%5D.ttf"),
    ("Cantarell", "ofl", "cantarell", "Cantarell-Regular.ttf"),
    ("Changa", "ofl", "changa", "Changa%5Bwght%5D.ttf"),
    ("Chakra Petch", "ofl", "chakrapetch", "ChakraPetch-Regular.ttf"),
    ("Chivo", "ofl", "chivo", "Chivo%5Bwght%5D.ttf"),
    ("Comfortaa", "ofl", "comfortaa", "Comfortaa%5Bwght%5D.ttf"),
    ("Cuprum", "ofl", "cuprum", "Cuprum%5Bwght%5D.ttf"),
    ("Didact Gothic", "ofl", "didactgothic", "DidactGothic-Regular.ttf"),
    ("Economica", "ofl", "economica", "Economica-Regular.ttf"),
    ("Electrolize", "ofl", "electrolize", "Electrolize-Regular.ttf"),
    ("Encode Sans", "ofl", "encodesans", "EncodeSans%5Bwdth,wght%5D.ttf"),
    ("Faustina", "ofl", "faustina", "Faustina%5Bwght%5D.ttf"),
    ("Fira Code", "ofl", "firacode", "FiraCode%5Bwght%5D.ttf"),
    ("Fjalla One", "ofl", "fjallaone", "FjallaOne-Regular.ttf"),
    ("Francois One", "ofl", "francoisone", "FrancoisOne-Regular.ttf"),
    ("Fredoka", "ofl", "fredoka", "Fredoka%5Bwdth,wght%5D.ttf"),
    ("Gothic A1", "ofl", "gothica1", "GothicA1-Regular.ttf"),
    ("Gruppo", "ofl", "gruppo", "Gruppo-Regular.ttf"),
    ("Heebo", "ofl", "heebo", "Heebo%5Bwght%5D.ttf"),
    ("Hind Madurai", "ofl", "hindmadurai", "HindMadurai-Regular.ttf"),
    ("Hind Siliguri", "ofl", "hindsiliguri", "HindSiliguri-Regular.ttf"),
    ("Homenaje", "ofl", "homenaje", "Homenaje-Regular.ttf"),
    ("IBM Plex Sans", "ofl", "ibmplexsans", "IBMPlexSans-Regular.ttf"),
    ("Istok Web", "ofl", "istokweb", "IstokWeb-Regular.ttf"),
    ("Jaldi", "ofl", "jaldi", "Jaldi-Regular.ttf"),
    ("Jost", "ofl", "jost", "Jost%5Bwght%5D.ttf"),
    ("Khand", "ofl", "khand", "Khand-Regular.ttf"),
    ("Kodchasan", "ofl", "kodchasan", "Kodchasan-Regular.ttf"),
    ("Krub", "ofl", "krub", "Krub-Regular.ttf"),
    ("Lato Bold", "ofl", "lato", "Lato-Bold.ttf"),
    ("Lekton", "ofl", "lekton", "Lekton-Regular.ttf"),
    ("Libre Franklin", "ofl", "librefranklin", "LibreFranklin%5Bwght%5D.ttf"),
    ("Lilita One", "ofl", "lilitaone", "LilitaOne-Regular.ttf"),
    ("Livvic", "ofl", "livvic", "Livvic-Regular.ttf"),
    ("Major Mono Display", "ofl", "majormonodisplay", "MajorMonoDisplay-Regular.ttf"),
    ("Michroma", "ofl", "michroma", "Michroma-Regular.ttf"),
    ("Monda", "ofl", "monda", "Monda-Regular.ttf"),
    ("Mplus 1p", "ofl", "mplus1p", "Mplus1p-Regular.ttf"),
    ("Nanum Gothic", "ofl", "nanumgothic", "NanumGothic-Regular.ttf"),
    ("Nobile", "ofl", "nobile", "Nobile-Regular.ttf"),
    ("Noto Sans Mono", "ofl", "notosansmono", "NotoSansMono%5Bwdth,wght%5D.ttf"),
    ("Offside", "ofl", "offside", "Offside-Regular.ttf"),
    ("Oxanium", "ofl", "oxanium", "Oxanium%5Bwght%5D.ttf"),
    ("Paytone One", "ofl", "paytoneone", "PaytoneOne-Regular.ttf"),
    ("Play Bold", "ofl", "play", "Play-Bold.ttf"),
    ("Pontano Sans", "ofl", "pontanosans", "PontanoSans%5Bwght%5D.ttf"),
    ("Rajdhani", "ofl", "rajdhani", "Rajdhani-Regular.ttf"),
    ("Rambla", "ofl", "rambla", "Rambla-Regular.ttf"),
    ("Red Hat Display", "ofl", "redhatdisplay", "RedHatDisplay%5Bwght%5D.ttf"),
    ("Red Hat Mono", "ofl", "redhatmono", "RedHatMono%5Bwght%5D.ttf"),
    ("Red Hat Text", "ofl", "redhattext", "RedHatText%5Bwght%5D.ttf"),
    ("Righteous", "ofl", "righteous", "Righteous-Regular.ttf"),
    ("Roboto Condensed", "apache", "robotocondensed", "static/RobotoCondensed-Regular.ttf"),
    ("Roboto Slab", "apache", "robotoslab", "static/RobotoSlab-Regular.ttf"),
    ("Rozha One", "ofl", "rozhaone", "RozhaOne-Regular.ttf"),
    ("Sarabun", "ofl", "sarabun", "Sarabun-Regular.ttf"),
    ("Sarpanch", "ofl", "sarpanch", "Sarpanch-Regular.ttf"),
    ("Secular One", "ofl", "secularone", "SecularOne-Regular.ttf"),
    ("Signika Negative", "ofl", "signikanegative", "SignikaNegative%5Bwght%5D.ttf"),
    ("Sintony", "ofl", "sintony", "Sintony-Regular.ttf"),
    ("Sora", "ofl", "sora", "Sora%5Bwght%5D.ttf"),
    ("Space Grotesk", "ofl", "spacegrotesk", "SpaceGrotesk%5Bwght%5D.ttf"),
    ("Spectral", "ofl", "spectral", "Spectral-Regular.ttf"),
    ("Squada One", "ofl", "squadaone", "SquadaOne-Regular.ttf"),
    ("Stint Ultra Expanded", "ofl", "stintultraexpanded", "StintUltraExpanded-Regular.ttf"),
    ("Syncopate", "ofl", "syncopate", "Syncopate-Regular.ttf"),
    ("Tajawal", "ofl", "tajawal", "Tajawal-Regular.ttf"),
    ("Tektur", "ofl", "tektur", "Tektur%5Bwdth,wght%5D.ttf"),
    ("Tenor Sans", "ofl", "tenorsans", "TenorSans-Regular.ttf"),
    ("Tinos", "apache", "tinos", "Tinos-Regular.ttf"),
    ("Titan One", "ofl", "titanone", "TitanOne-Regular.ttf"),
    ("Trirong", "ofl", "trirong", "Trirong-Regular.ttf"),
    ("Ubuntu Bold", "ubuntu", "ubuntu", "Ubuntu-Bold.ttf"),
    ("VT323", "ofl", "vt323", "VT323-Regular.ttf"),
    ("Wallpoet", "ofl", "wallpoet", "Wallpoet-Regular.ttf"),
    ("Wire One", "ofl", "wireone", "WireOne-Regular.ttf"),
    ("Yantramanav", "ofl", "yantramanav", "Yantramanav-Regular.ttf"),
    ("Zen Dots", "ofl", "zendots", "ZenDots-Regular.ttf"),
    ("Zilla Slab", "ofl", "zillaslab", "ZillaSlab-Regular.ttf"),
]


def _download_google_fonts(cache_dir: Path) -> list[Path]:
    """Download curated Google Fonts TTFs to ``cache_dir``.

    Returns the list of successfully-cached TTF file paths. Missing
    files are skipped; network errors per-font do not abort the set.
    """
    try:
        import requests
    except ImportError:
        print("[google-fonts] requests not installed", file=sys.stderr)
        return []

    cache_dir.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/google/fonts/main"
    paths: list[Path] = []
    to_download: list[tuple[str, str, Path]] = []
    for display, lic, folder, ttf in _GOOGLE_FONTS:
        # Local filename: flatten "static/X.ttf" into a unique name.
        safe = ttf.replace("/", "_").replace("%5B", "[").replace("%5D", "]")
        local = cache_dir / f"{folder}__{safe}"
        if local.exists() and local.stat().st_size > 1024:
            paths.append(local)
            continue
        url = f"{base}/{lic}/{folder}/{ttf}"
        to_download.append((display, url, local))

    if to_download:
        print(f"[google-fonts] downloading {len(to_download)} fonts "
              f"({len(paths)} already cached)")
    ok = 0
    fail = 0
    for display, url, local in to_download:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 1024:
                local.write_bytes(r.content)
                paths.append(local)
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    if to_download:
        print(f"[google-fonts] downloaded {ok} ok, {fail} failed, "
              f"total cached: {len(paths)}")
    return paths


def _google_fonts_stream(max_samples: int = 400_000) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (image, label) pairs synthesized from ~150 Google Fonts.

    Mirrors ``_mjsynth_stream``'s label mix but swaps the 15 Windows
    system fonts for a curated Google Fonts subset (sans-serif +
    mono). TTFs are cached under
    ``%TEMP%/crnn_pretrain_cache/google_fonts/`` and deleted when
    the generator is exhausted.
    """
    from PIL import ImageDraw, ImageFont, ImageFilter

    cache_dir = Path(tempfile.gettempdir()) / "crnn_pretrain_cache" / "google_fonts"
    ttf_paths = _download_google_fonts(cache_dir)
    if not ttf_paths:
        print("[google-fonts] no fonts available, skipping", file=sys.stderr)
        return

    rng = random.Random(13)
    fonts: list[ImageFont.FreeTypeFont] = []
    for p in ttf_paths:
        for sz in (18, 22, 26, 30):
            try:
                fonts.append(ImageFont.truetype(str(p), size=sz))
            except Exception:
                pass
    if not fonts:
        print("[google-fonts] no usable fonts parsed", file=sys.stderr)
        return

    # Re-use the SC vocabulary from _mjsynth_stream's body.
    _SC_VOCAB = [
        "IRON", "COPPER", "TITANIUM", "QUANTANIUM", "LARANITE",
        "BERYL", "TARANITE", "BORASE", "HEPHAESTANITE", "AGRICIUM",
        "GOLD", "TIN", "ALUMINUM", "TUNGSTEN", "CORUNDUM", "DIAMOND",
        "BEXALITE", "QUARTZ", "ORE", "RAW", "ICE", "CRYSTAL", "GEM",
        "IRON (ORE)", "COPPER (ORE)", "RAW ICE", "QUANTANIUM (RAW)",
        "MASS", "RESISTANCE", "INSTABILITY", "COMPOSITION",
        "SCAN RESULTS", "EASY", "MEDIUM", "HARD", "EXTREME",
    ]

    print(f"[google-fonts] streaming up to {max_samples} samples "
          f"({len(ttf_paths)} families, {len(fonts)} font variants)")
    count = 0
    try:
        while count < max_samples:
            r = rng.random()
            if r < 0.25:
                # Plain integer
                n = rng.choices([1, 2, 3, 4, 5, 6, 7],
                                weights=[5, 12, 28, 25, 15, 10, 5])[0]
                label = "".join(rng.choice("0123456789") for _ in range(n))
            elif r < 0.40:
                # Decimal
                whole = rng.choices([1, 2, 3], weights=[50, 35, 15])[0]
                frac = rng.choices([1, 2], weights=[40, 60])[0]
                label = ("".join(rng.choice("0123456789") for _ in range(whole))
                         + "."
                         + "".join(rng.choice("0123456789") for _ in range(frac)))
            elif r < 0.55:
                # Percentage
                n = rng.choices([1, 2, 3], weights=[35, 60, 5])[0]
                if n == 3:
                    label = "100%"
                else:
                    label = "".join(rng.choice("0123456789") for _ in range(n)) + "%"
            elif r < 0.80:
                # SC vocabulary token
                label = rng.choice(_SC_VOCAB)
            else:
                # Random uppercase word
                n = rng.choices([2, 3, 4, 5, 6, 7, 8],
                                weights=[5, 15, 25, 25, 15, 10, 5])[0]
                label = "".join(
                    rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n)
                )

            if not _label_ok(label):
                continue

            font = rng.choice(fonts)
            canvas_w = 200
            img = Image.new("L", (canvas_w, CANVAS_H + 12), color=0)
            draw = ImageDraw.Draw(img)
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                bbox = (0, 0, len(label) * 10, 16)
                tw, th = bbox[2], bbox[3]
            tx = (canvas_w - tw) // 2 - bbox[0]
            ty = (CANVAS_H + 12 - th) // 2 - bbox[1]
            try:
                draw.text((tx, ty), label,
                          fill=rng.randint(200, 255), font=font)
            except Exception:
                continue

            # Augmentations — same mix as _mjsynth_stream
            if rng.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
            if rng.random() < 0.2:
                img = img.rotate(rng.uniform(-2, 2),
                                 resample=Image.BILINEAR, fillcolor=0)

            arr = np.array(img, dtype=np.uint8)
            mask = arr > 100
            if not mask.any():
                continue
            ys = np.where(mask.any(axis=1))[0]
            xs = np.where(mask.any(axis=0))[0]
            if len(ys) < 5 or len(xs) < 3:
                continue
            pad = 3
            y1 = max(0, ys[0] - pad)
            y2 = min(arr.shape[0], ys[-1] + pad + 1)
            x1 = max(0, xs[0] - pad)
            x2 = min(arr.shape[1], xs[-1] + pad + 1)
            crop = arr[y1:y2, x1:x2]
            yield _normalize_image(Image.fromarray(crop)), label
            count += 1
    finally:
        # Clean up downloaded TTFs on generator exhaustion / close.
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[google-fonts] deleted cache {cache_dir}")
        except OSError:
            pass


# ── SynthText streamer ─────────────────────────────────────────────


def _emnist_stream(max_samples: int = 200_000) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (image, label) pairs from EMNIST ByClass (814 K letters+digits).

    EMNIST is NIST's handwritten character set, split across 62
    classes (0-9, A-Z, a-z). It's the de-facto "MNIST with letters"
    dataset. We stream the 'byclass' subset, yield one character
    per sample, and delete the download after iteration.

    Works best when combined with MJSynth-synth (diverse fonts) —
    EMNIST covers handwritten shape variance, MJSynth-synth covers
    font variance. Together they approximate the real-world digit+
    letter distribution Paddle learned from.
    """
    try:
        import requests
    except ImportError:
        print("[emnist] requests not installed", file=sys.stderr)
        return

    cache_dir = Path(tempfile.gettempdir()) / "crnn_pretrain_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # NIST's official EMNIST mirror is slow; use the
    # community-maintained torchvision-hosted npz mirror instead.
    # Contains digit + letter labeled classes; we filter to our alphabet.
    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    # If that's unreachable we fall back to the alternative:
    # we'll silently return and move on (MJSynth-synth already covers
    # letter variety).
    zip_path = cache_dir / "emnist.zip"
    if not zip_path.exists():
        print(f"[emnist] downloading {url} (~1.1 GB)...", file=sys.stderr)
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                total = 0
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
                            if total % (32 << 20) < (1 << 20):
                                print(f"[emnist] {total/1e6:.0f} MB...", file=sys.stderr)
            print(f"[emnist] downloaded {total/1e6:.1f} MB", file=sys.stderr)
        except Exception as exc:
            print(f"[emnist] download failed: {exc}", file=sys.stderr)
            try:
                zip_path.unlink()
            except OSError:
                pass
            return

    # Extract and parse the ByClass subset (small IDX-format binaries)
    # Packaged as: EMNIST/gzip/emnist-byclass-train-images-idx3-ubyte.gz
    import zipfile
    import gzip
    import struct
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            img_name = next(
                (n for n in names if "byclass-train-images" in n and n.endswith(".gz")),
                None,
            )
            lbl_name = next(
                (n for n in names if "byclass-train-labels" in n and n.endswith(".gz")),
                None,
            )
            if not img_name or not lbl_name:
                print(f"[emnist] byclass files not in zip; names: {names[:5]}", file=sys.stderr)
                return
            with zf.open(img_name) as fraw:
                img_bytes = gzip.decompress(fraw.read())
            with zf.open(lbl_name) as fraw:
                lbl_bytes = gzip.decompress(fraw.read())
    except Exception as exc:
        print(f"[emnist] zip parse failed: {exc}", file=sys.stderr)
        return

    # IDX format: magic (4), count (4), rows (4), cols (4), then pixels
    _, n_imgs, rows, cols = struct.unpack(">IIII", img_bytes[:16])
    _, n_lbls = struct.unpack(">II", lbl_bytes[:8])
    n = min(n_imgs, n_lbls, max_samples)
    print(f"[emnist] streaming {n} samples ({rows}x{cols})", file=sys.stderr)

    # EMNIST ByClass label mapping: 0-9 = digits, 10-35 = A-Z, 36-61 = a-z
    def _idx_to_char(i: int) -> Optional[str]:
        if 0 <= i <= 9:
            return str(i)
        if 10 <= i <= 35:
            return chr(ord("A") + i - 10)
        if 36 <= i <= 61:
            return chr(ord("a") + i - 36)
        return None

    pixels_per_img = rows * cols
    img_data_start = 16
    lbl_data_start = 8
    for i in range(n):
        lbl_idx = lbl_bytes[lbl_data_start + i]
        ch = _idx_to_char(int(lbl_idx))
        if ch is None or ch not in ALLOWED_CHARS:
            continue
        px = img_bytes[img_data_start + i * pixels_per_img
                       : img_data_start + (i + 1) * pixels_per_img]
        arr = np.frombuffer(px, dtype=np.uint8).reshape(rows, cols)
        # EMNIST is transposed; fix to canonical orientation.
        arr = arr.T
        yield _normalize_image(Image.fromarray(arr)), ch

    try:
        zip_path.unlink()
        print(f"[emnist] deleted cache {zip_path}", file=sys.stderr)
    except OSError:
        pass


# ── TextOCR streamer ───────────────────────────────────────────────


def _textocr_stream(max_samples: int = 300_000) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (image, label) pairs from Facebook's TextOCR dataset.

    TextOCR contains ~900 K word-level crops from natural OpenImages
    photos with GT text labels. The full image corpus is ~7 GB so we
    stream per-example from a HuggingFace mirror of the pre-cropped
    word subset (``MiXaiLL76/TextOCR_OCR``) via ``datasets`` with
    ``streaming=True`` — individual parquet shards are fetched on
    demand into our pretrain cache and that cache is deleted when
    the generator exits. Labels are filtered to our alphabet and
    dropped if longer than 12 chars, matching the other streamers.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[textocr] pip install datasets", file=sys.stderr)
        return

    # Point HF's caches into our own pretrain cache dir so cleanup is
    # centralised with the other streamers.
    cache_dir = Path(tempfile.gettempdir()) / "crnn_pretrain_cache" / "textocr_hf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prev_hf_home = os.environ.get("HF_HOME")
    os.environ["HF_HOME"] = str(cache_dir)

    try:
        try:
            # streaming=False: download all parquet shards to the
            # local cache once, then iterate from disk. Much faster
            # than per-batch byte-range requests (observed ~50 batches
            # per 10 minutes with streaming vs ~5000 batches per minute
            # once the shards are local).
            print(f"[textocr] bulk-downloading parquet shards (~1-2 GB) to {cache_dir}",
                  file=sys.stderr)
            ds = load_dataset(
                "MiXaiLL76/TextOCR_OCR",
                split="train",
                streaming=False,
                cache_dir=str(cache_dir),
            )
        except Exception as exc:
            print(f"[textocr] load_dataset failed: {exc}", file=sys.stderr)
            return

        print(f"[textocr] iterating up to {max_samples} samples from local cache "
              f"(MiXaiLL76/TextOCR_OCR, {len(ds)} total)", file=sys.stderr)
        seen = 0
        kept = 0
        try:
            for ex in ds:
                if seen >= max_samples:
                    break
                seen += 1
                label = ex.get("text")
                img = ex.get("image")
                if not isinstance(label, str) or img is None:
                    continue
                label = label.strip()
                if not _label_ok(label):
                    continue
                if not isinstance(img, Image.Image):
                    try:
                        if isinstance(img, (bytes, bytearray)):
                            img = Image.open(io.BytesIO(img))
                        else:
                            img = Image.fromarray(np.asarray(img))
                    except Exception:
                        continue
                try:
                    arr = _normalize_image(img)
                except Exception:
                    continue
                if arr.shape[0] != CANVAS_H or arr.shape[1] < 8:
                    continue
                kept += 1
                yield arr, label
            print(f"[textocr] yielded {kept}/{seen} examples (alphabet-filtered)",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[textocr] stream interrupted: {exc}", file=sys.stderr)
    finally:
        # Restore HF_HOME and delete the per-run shard cache.
        if prev_hf_home is None:
            os.environ.pop("HF_HOME", None)
        else:
            os.environ["HF_HOME"] = prev_hf_home
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[textocr] deleted cache {cache_dir}", file=sys.stderr)
        except OSError:
            pass


def _cocotext_stream(max_samples: int = 150_000) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (image, label) pairs from COCO-Text v2 word annotations.

    COCO-Text is ~170 K labeled word polygons over the MS COCO
    train2014 images. We download just the annotations JSON
    (~30 MB) and then fetch images on-demand from the COCO CDN,
    crop each word polygon, and emit (normalized_image, label).
    Images are downloaded one-at-a-time and discarded — total
    on-disk footprint stays under ~5 MB at any time.

    Falls back to HuggingFace streaming if ``datasets`` is
    installed and hosts ``coco-text``. If neither the HF nor
    direct path is reachable, silently yields nothing.
    """
    # ── HF streaming path (preferred if available) ──────────────
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        load_dataset = None  # type: ignore
    if load_dataset is not None:
        for repo in ("coco-text", "yuchenlin/coco-text", "nlphuji/coco-text"):
            try:
                ds = load_dataset(repo, split="train", streaming=True)
                print(f"[cocotext] HF streaming from '{repo}'", file=sys.stderr)
                count = 0
                for ex in ds:
                    if count >= max_samples:
                        break
                    # Be liberal about field names
                    img = ex.get("image") or ex.get("img")
                    label = (ex.get("text") or ex.get("label")
                             or ex.get("utf8_string") or "")
                    legibility = ex.get("legibility", "legible")
                    if legibility and str(legibility).lower() != "legible":
                        continue
                    if not isinstance(label, str) or not _label_ok(label):
                        continue
                    if img is None:
                        continue
                    if not isinstance(img, Image.Image):
                        try:
                            img = Image.open(io.BytesIO(img)) if isinstance(img, (bytes, bytearray)) else Image.fromarray(np.asarray(img))
                        except Exception:
                            continue
                    try:
                        yield _normalize_image(img), label
                        count += 1
                    except Exception:
                        continue
                if count > 0:
                    return
            except Exception as exc:
                print(f"[cocotext] HF '{repo}' failed: {exc}", file=sys.stderr)
                continue

    # ── Direct path: annotations JSON + on-demand image fetch ───
    try:
        import requests
    except ImportError:
        print("[cocotext] requests not installed", file=sys.stderr)
        return

    cache_dir = Path(tempfile.gettempdir()) / "crnn_pretrain_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    anno_path = cache_dir / "cocotext.v2.json"

    if not anno_path.exists():
        # Official COCO-Text v2 annotations (~30 MB). Mirror chain:
        candidates = [
            "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip",
            "https://vision.cornell.edu/se3/wp-content/uploads/2019/01/cocotext.v2.zip",
        ]
        zip_path = cache_dir / "cocotext.v2.zip"
        downloaded = False
        for url in candidates:
            try:
                print(f"[cocotext] downloading {url}", file=sys.stderr)
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    total = 0
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                                total += len(chunk)
                    print(f"[cocotext] downloaded {total/1e6:.1f} MB", file=sys.stderr)
                downloaded = True
                break
            except Exception as exc:
                print(f"[cocotext] {url} failed: {exc}", file=sys.stderr)
                zip_path.unlink(missing_ok=True)

        if not downloaded:
            print("[cocotext] no mirror reachable; skipping", file=sys.stderr)
            return

        import zipfile
        try:
            with zipfile.ZipFile(zip_path) as zf:
                json_name = next(
                    (n for n in zf.namelist() if n.endswith(".json")),
                    None,
                )
                if not json_name:
                    print("[cocotext] no .json in zip", file=sys.stderr)
                    zip_path.unlink(missing_ok=True)
                    return
                with zf.open(json_name) as fraw, open(anno_path, "wb") as fout:
                    shutil.copyfileobj(fraw, fout)
        except Exception as exc:
            print(f"[cocotext] zip extract failed: {exc}", file=sys.stderr)
            zip_path.unlink(missing_ok=True)
            return
        zip_path.unlink(missing_ok=True)

    import json
    try:
        with open(anno_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[cocotext] JSON parse failed: {exc}", file=sys.stderr)
        anno_path.unlink(missing_ok=True)
        return

    imgs = data.get("imgs", {})   # id -> {file_name, ...}
    anns = data.get("anns", {})   # id -> {image_id, utf8_string, bbox, legibility}
    print(f"[cocotext] {len(anns)} annotations, {len(imgs)} images", file=sys.stderr)

    # Group annotations by image to minimize redundant image downloads.
    from collections import defaultdict
    by_img: dict = defaultdict(list)
    for a in anns.values():
        if not isinstance(a, dict):
            continue
        if a.get("legibility", "legible") != "legible":
            continue
        lbl = a.get("utf8_string", "")
        if not isinstance(lbl, str) or not _label_ok(lbl):
            continue
        by_img[a.get("image_id")].append(a)

    emitted = 0
    rng = random.Random(11)
    img_ids = list(by_img.keys())
    rng.shuffle(img_ids)

    sess = requests.Session()
    for img_id in img_ids:
        if emitted >= max_samples:
            break
        meta = imgs.get(str(img_id)) or imgs.get(img_id)
        if not meta:
            continue
        file_name = meta.get("file_name") or ""
        # COCO train2014 images live at this CDN path.
        url = f"https://images.cocodataset.org/train2014/{file_name}"
        try:
            resp = sess.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            full_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            continue

        W, H = full_img.size
        for a in by_img[img_id]:
            if emitted >= max_samples:
                break
            lbl = a.get("utf8_string", "")
            bbox = a.get("bbox") or []
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(W, int(round(x + w)))
            y2 = min(H, int(round(y + h)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            try:
                crop = full_img.crop((x1, y1, x2, y2))
                yield _normalize_image(crop), lbl
                emitted += 1
            except Exception:
                continue

    try:
        anno_path.unlink()
        print(f"[cocotext] deleted cache {anno_path}", file=sys.stderr)
    except OSError:
        pass


def _synthtext_stream(max_samples: int = 200_000) -> Iterator[tuple[np.ndarray, str]]:
    """Yield (image, label) from SynthText's word-crop subset.

    The full SynthText dataset is ~41 GB — we never touch that.
    Instead we try HuggingFace mirrors that host a pre-cropped
    word-level subset (~1 GB streamed in shards). If no mirror
    is reachable or ``datasets`` is not installed, logs "not
    available" and returns cleanly so pretraining continues.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[synthtext] datasets package not installed; not available", file=sys.stderr)
        return

    # Try a chain of known mirrors. Each is a word-crop subset —
    # NOT the 41 GB full release.
    candidates = [
        ("sergiopaniego/SynthText", "train"),
        ("priyank-m/SynthText_Eng_word_level", "train"),
        ("priyank-m/SynthText_word_crops", "train"),
        ("lmms-lab/SynthText", "train"),
    ]
    ds = None
    chosen = None
    for repo, split in candidates:
        try:
            ds = load_dataset(repo, split=split, streaming=True)
            chosen = repo
            print(f"[synthtext] HF streaming from '{repo}'", file=sys.stderr)
            break
        except Exception as exc:
            print(f"[synthtext] '{repo}' unavailable: {exc}", file=sys.stderr)
            continue

    if ds is None:
        print("[synthtext] not available (all mirrors 403/404)", file=sys.stderr)
        return

    count = 0
    try:
        for ex in ds:
            if count >= max_samples:
                break
            img = (ex.get("image") or ex.get("img") or ex.get("jpg"))
            label = (ex.get("text") or ex.get("label")
                     or ex.get("word") or ex.get("txt") or "")
            if not isinstance(label, str) or not _label_ok(label):
                continue
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                try:
                    if isinstance(img, (bytes, bytearray)):
                        img = Image.open(io.BytesIO(img))
                    else:
                        img = Image.fromarray(np.asarray(img))
                except Exception:
                    continue
            try:
                yield _normalize_image(img), label
                count += 1
            except Exception:
                continue
    except Exception as exc:
        print(f"[synthtext] stream interrupted ({chosen}): {exc}", file=sys.stderr)
        return


# ── Streaming dataset adapter ──────────────────────────────────────


class _IterableStreamDataset:
    """Wraps a Python generator so it plays with torch DataLoader.

    Buffers up to ``buffer_size`` items then shuffles each chunk
    before yielding — approximates uniform shuffling of a huge
    stream without materialising it all in memory.
    """
    def __init__(self, generator, buffer_size: int = 2048):
        self.gen = generator
        self.buffer_size = buffer_size

    def __iter__(self):
        buf: list[tuple[np.ndarray, str]] = []
        rng = random.Random(0)
        for sample in self.gen:
            buf.append(sample)
            if len(buf) >= self.buffer_size:
                rng.shuffle(buf)
                for s in buf:
                    yield s
                buf.clear()
        rng.shuffle(buf)
        yield from buf


# ── CRNN model (shared w/ train_crnn.py) ───────────────────────────


def build_model(size: str = "small"):
    """Pretrain's CRNN mirrors train_crnn's. ``size`` selects capacity:
    'small' (1.3M, legacy) or 'large' (~5M)."""
    import torch
    import torch.nn as nn

    if size == "small":
        c = (32, 64, 128, 256, 256)
        rnn_hidden = 128
    elif size == "large":
        c = (48, 96, 192, 384, 384)
        rnn_hidden = 256
    else:
        raise ValueError(f"unknown size {size!r}")

    class CRNN(nn.Module):
        def __init__(self, n_tokens: int = NUM_TOKENS):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, c[0], 3, padding=1), nn.BatchNorm2d(c[0]), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(c[0], c[1], 3, padding=1), nn.BatchNorm2d(c[1]), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(c[1], c[2], 3, padding=1), nn.BatchNorm2d(c[2]), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(c[2], c[3], 3, padding=1), nn.BatchNorm2d(c[3]), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(c[3], c[4], (2, 1)), nn.BatchNorm2d(c[4]), nn.ReLU(),
            )
            self.rnn = nn.LSTM(
                input_size=c[4], hidden_size=rnn_hidden,
                num_layers=2, bidirectional=True, dropout=0.1,
            )
            self.fc = nn.Linear(rnn_hidden * 2, n_tokens)

        def forward(self, x):
            f = self.cnn(x)
            B, C, H, T = f.shape
            if H != 1:
                raise RuntimeError(f"expected H=1, got {H}")
            f = f.squeeze(2).permute(2, 0, 1)
            r, _ = self.rnn(f)
            return self.fc(r)
    return CRNN()


def _label_to_indices(label: str) -> list[int]:
    return [CHAR_CLASSES.index(c) for c in label if c in CHAR_CLASSES]


# ── Training loop ──────────────────────────────────────────────────


def _collate(batch):
    import torch
    imgs, targets, labels = zip(*batch)
    max_w = max(t.shape[-1] for t in imgs)
    B = len(imgs)
    padded = torch.zeros((B, 1, CANVAS_H, max_w), dtype=torch.float32)
    for i, t in enumerate(imgs):
        padded[i, :, :, :t.shape[-1]] = t
    target_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_flat = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    input_lens = torch.tensor(
        [max(1, img.shape[-1] // 4 - 1) for img in imgs], dtype=torch.long,
    )
    return padded, targets_flat, input_lens, target_lens, list(labels)


def _batch_stream(gen, batch_size: int):
    import torch
    batch: list[tuple[np.ndarray, str]] = []
    for img, label in gen:
        idx = _label_to_indices(label)
        if not idx:
            continue
        t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        target = torch.tensor(idx, dtype=torch.long)
        batch.append((t, target, label))
        if len(batch) >= batch_size:
            yield _collate(batch)
            batch.clear()
    if batch:
        yield _collate(batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="svhn,mjsynth-synth",
                        help="Comma-separated: svhn, mjsynth-synth, synthtext")
    parser.add_argument("--epochs", type=int, default=2, help="Passes over the stream")
    parser.add_argument("--max-per", type=int, default=60000,
                        help="Max samples to draw per dataset per epoch")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default=str(_OUT_PT),
                        help="PyTorch state_dict output path")
    parser.add_argument("--size", type=str, default="small",
                        choices=("small", "large"),
                        help="Model capacity: 'small' (1.3M) or 'large' (~5M)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("ERROR: pip install torch onnx", file=sys.stderr)
        sys.exit(1)

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    stream_fns = {
        "svhn": _svhn_stream,
        "mjsynth-synth": _mjsynth_stream,
        "mjsynth": _mjsynth_stream,   # alias
        "google-fonts": _google_fonts_stream,
        "gfonts": _google_fonts_stream,   # alias
        "emnist": _emnist_stream,
        "synthtext": _synthtext_stream,
        "cocotext": _cocotext_stream,
        "coco-text": _cocotext_stream,  # alias
        "textocr": _textocr_stream,
        "furore": _furore_stream,  # SC mining HUD font (primary)
    }
    for d in datasets:
        if d not in stream_fns:
            print(f"Unknown dataset: {d}", file=sys.stderr)
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(size=args.size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters on {device}")

    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"--- Pretrain epoch {epoch+1}/{args.epochs} ---")
        for ds_name in datasets:
            gen = stream_fns[ds_name](max_samples=args.max_per)
            if gen is None:
                continue
            model.train()
            loss_sum = 0.0
            n_batches = 0
            for padded, targets_flat, input_lens, target_lens, _labels in _batch_stream(gen, args.batch_size):
                padded = padded.to(device)
                optimizer.zero_grad()
                logits = model(padded)
                log_probs = logits.log_softmax(-1).cpu()
                T = log_probs.shape[0]
                input_lens_c = input_lens.clamp(max=T)
                loss = criterion(log_probs, targets_flat, input_lens_c, target_lens)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                loss_sum += loss.item()
                n_batches += 1
                if n_batches % 50 == 0:
                    print(f"  [{ds_name}] batch {n_batches}  loss={loss_sum/n_batches:.3f}")
            if n_batches:
                avg = loss_sum / n_batches
                print(f"  [{ds_name}] epoch avg loss = {avg:.3f} over {n_batches} batches")
                if avg < best_loss:
                    best_loss = avg
                    torch.save(model.state_dict(), args.out)
                    print(f"  checkpoint: {args.out}")

    print(f"\nBest loss: {best_loss:.3f}")
    print(f"Weights: {args.out}")
    print("To fine-tune on SC data:")
    print(f"  python -m ocr.train_crnn --init-from {args.out} --epochs 15")

    # Aggressively clean the cache
    cache_dir = Path(tempfile.gettempdir()) / "crnn_pretrain_cache"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Cleaned cache {cache_dir}")


if __name__ == "__main__":
    if __package__ is None:
        sys.path.insert(0, str(_MODULE_DIR.parent))
    main()
