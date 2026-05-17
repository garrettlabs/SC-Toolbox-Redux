"""Trainer for the GRAYSCALE per-glyph mining-HUD CNN.

This is the HUD twin of ``train_signal_rgb_v2.py`` — same 3-conv-block
architecture, but:
  * Single-channel grayscale input: ``(N, 1, 28, 28)`` (the runtime's
    HUD per-glyph segmenter feeds 28×28 ``L`` mode tiles padded with
    255 around the inked bounding box).
  * 12-class output covering the full HUD alphabet: ``0123456789.%``.
    Folder-name ``dot`` ↔ ``.`` and ``pct`` ↔ ``%`` on disk (filesystem
    can't use ``.`` or ``%`` as a directory name without escaping).

**HUD font ≠ Signature font.** This trainer must NEVER load samples
from the signature staging dirs. To enforce that structurally:
  * The spec for ``"hud"`` in ``ocr.training_registry`` declares the
    only allowed source directories.
  * Every PNG path is run through ``assert_path_belongs_to("hud", ...)``
    before being loaded. A misconfigured cwd / wrong staging dir
    raises RegistryError instead of silently mixing fonts.
  * The output ONNX path comes from ``spec.model_path`` so a typo in
    the trainer can't overwrite the signature model.

Inputs:
  * Digits 0-9: ``training_data_user_panel/{0,1,...,9}/*.png``
  * Decimal point: ``training_data_user_panel/dot/*.png``
  * Percent: ``training_data_user_panel/pct/*.png``

The staging dir is resolved via ``training_registry.resolve_staging_dir("hud")``
which prefers the WingmanAI install location (where the live review
tools maintain the up-to-date corpus) and falls back to the dev tree.

Output (path from ``spec.model_path``):
  * ``ocr/models/model_hud_cnn.onnx``      — exported ONNX, opset 13
  * ``ocr/models/model_hud_cnn.json``      — alphabet / val metrics
  * ``ocr/models/model_hud_cnn_train.log`` — training log

Run::

    %LOCALAPPDATA%\\Python\\pythoncore-3.14-64\\python.exe \\
        ocr/train_hud_cnn.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance


# ── Paths via registry ──────────────────────────────────────────────

THIS_FILE = Path(__file__).resolve()
TOOL_DIR = THIS_FILE.parent.parent       # ...\Mining_Signals\
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from ocr.training_registry import (        # noqa: E402
    get as _registry_get,
    resolve_staging_dir,
    assert_path_belongs_to,
)

REGION_KIND = "hud"
SPEC = _registry_get(REGION_KIND)

CHAR_CLASSES = SPEC.label_set              # "0123456789.%" — 12 chars
assert CHAR_CLASSES == "0123456789.%", (
    f"HUD label set drift — spec says {CHAR_CLASSES!r}, expected "
    f"'0123456789.%' (12 chars). Refusing to train against a different "
    f"alphabet."
)
NUM_CLASSES = len(CHAR_CLASSES)             # 12

# Filesystem-safe folder name → label char mapping. The reverse map
# is what disk → label uses; nothing on disk lives under "." / "%".
FOLDER_LABEL_MAP = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "dot": ".",
    "pct": "%",
}

# ── Output paths from registry ──────────────────────────────────────

OUT_ONNX = Path(SPEC.model_path)
OUT_META = OUT_ONNX.with_suffix(".json")
OUT_LOG = OUT_ONNX.with_name(OUT_ONNX.stem + "_train.log")
OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)


# ── Hyperparameters ─────────────────────────────────────────────────

EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3
VAL_SPLIT = 0.15
SEED = 1337
IMG_SIZE = 28

# Augmentation passes per real sample. Smallest class is class "9"
# with ~55 samples, so we want enough augmentation to give the model
# headroom. Same multiplier the v2 signal trainer uses on its real
# samples.
N_AUGS_PER_SAMPLE = 6

# Class floor — refuse to train if any class has fewer than this.
FLOOR_PER_CLASS = SPEC.floor_per_class       # 30


# ── Logging ─────────────────────────────────────────────────────────

log = logging.getLogger("train_hud_cnn")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(OUT_LOG, mode="w", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(message)s"))
log.handlers.clear()
log.addHandler(_fh)
log.addHandler(_sh)


# ── Augmentation utilities (grayscale, "L" mode) ────────────────────

def _affine_jitter(
    img: Image.Image, *,
    max_rot_deg: float,
    max_trans_frac: float,
    min_scale: float,
    max_scale: float,
    rng: random.Random,
) -> Image.Image:
    """Rotate + translate + scale jitter on an "L" mode PIL image.

    Fill colour is 255 (the HUD pad colour after the segmenter's
    pad-then-resize step) so the rotated edges blend into existing
    background. Matches the inference-time padding behaviour exactly.
    """
    angle = rng.uniform(-max_rot_deg, max_rot_deg)
    sx = rng.uniform(min_scale, max_scale)
    sy = rng.uniform(min_scale, max_scale)
    tx = rng.uniform(-max_trans_frac, max_trans_frac) * img.size[0]
    ty = rng.uniform(-max_trans_frac, max_trans_frac) * img.size[1]
    out = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
    nw = max(1, int(round(out.size[0] * sx)))
    nh = max(1, int(round(out.size[1] * sy)))
    out = out.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", img.size, 255)
    cx = (img.size[0] - nw) // 2 + int(round(tx))
    cy = (img.size[1] - nh) // 2 + int(round(ty))
    canvas.paste(out, (cx, cy))
    return canvas


def _photo_jitter(img: Image.Image, *, rng: random.Random) -> Image.Image:
    """Brightness + contrast jitter for grayscale crops."""
    b = rng.uniform(0.80, 1.20)
    c = rng.uniform(0.80, 1.20)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img


def _augment_sample(
    src: Image.Image, *, n: int, rng: random.Random,
) -> List[Image.Image]:
    """Generate ``n`` (including the original) augmented variants."""
    base = src.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    out: List[Image.Image] = [base]
    for _ in range(n - 1):
        a = _affine_jitter(
            base,
            max_rot_deg=8.0,
            max_trans_frac=0.06,
            min_scale=0.90,
            max_scale=1.05,
            rng=rng,
        )
        a = _photo_jitter(a, rng=rng)
        out.append(a)
    return out


# ── Data loading with HARD tripwire ─────────────────────────────────

def _enumerate_class_files(staging: Path) -> dict[str, List[Path]]:
    """For each label char, list its on-disk PNGs.

    Validates that EVERY path lives under one of the HUD spec's
    registered training source directories. Anything outside that set
    raises RegistryError before training begins — this is the
    structural HUD/Signature isolation guarantee.
    """
    out: dict[str, List[Path]] = {ch: [] for ch in CHAR_CLASSES}
    for folder_name, label_ch in FOLDER_LABEL_MAP.items():
        cls_dir = staging / folder_name
        if not cls_dir.is_dir():
            log.warning(
                "[load] class %r folder missing: %s", label_ch, cls_dir,
            )
            continue
        # Tripwire — ensure the directory itself is HUD-registered.
        assert_path_belongs_to(REGION_KIND, cls_dir)
        for png in sorted(cls_dir.glob("*.png")):
            # Tripwire — each individual file before it touches RAM.
            assert_path_belongs_to(REGION_KIND, png)
            out[label_ch].append(png)
    return out


def load_dataset(
    staging: Path, rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load + augment all HUD glyph crops.

    Returns ``(X, y, per_class_counts)``:
      X: (N, 1, 28, 28) float32 in [0, 1], DARK-on-LIGHT (255 bg)
      y: (N,) int64 class index into ``CHAR_CLASSES``
      per_class_counts: ``{label_char: raw_sample_count}``
    """
    by_class = _enumerate_class_files(staging)
    log.info("[load] raw per-class file counts:")
    for ch in CHAR_CLASSES:
        log.info("  %r: %d files", ch, len(by_class[ch]))

    # Floor check.
    for ch in CHAR_CLASSES:
        if len(by_class[ch]) < FLOOR_PER_CLASS:
            raise RuntimeError(
                f"Class {ch!r} has {len(by_class[ch])} samples; "
                f"floor is {FLOOR_PER_CLASS}. Refusing to train an "
                f"unbalanced HUD CNN. Add more reviewed glyphs."
            )

    images: List[np.ndarray] = []
    labels: List[int] = []
    counts: dict = {}

    for cls_idx, ch in enumerate(CHAR_CLASSES):
        n_real = 0
        n_total = 0
        for src in by_class[ch]:
            try:
                base = Image.open(src).convert("L")
            except Exception as exc:
                log.warning("[load] skip %s: %s", src.name, exc)
                continue
            n_real += 1
            augs = _augment_sample(base, n=N_AUGS_PER_SAMPLE, rng=rng)
            for im in augs:
                arr = np.asarray(im, dtype=np.float32) / 255.0
                # (H, W) → (1, H, W)
                images.append(arr[None, :, :])
                labels.append(cls_idx)
                n_total += 1
        counts[ch] = n_real
        log.info(
            "[load] class %r: %d raw → %d augmented", ch, n_real, n_total,
        )

    X = np.stack(images, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, counts


# ── Model: 3 conv blocks + FC, grayscale input ──────────────────────

def build_model(num_classes: int):
    """Architecture matching ``train_signal_rgb_v2.py`` (3 conv blocks
    + FC head) but with single-channel input. First conv is
    ``Conv2d(1, 32, 3)`` — the only structural deviation from the RGB
    signature trainer."""
    import torch.nn as nn

    class HUDGrayCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),     # 28
                nn.ReLU(),
                nn.MaxPool2d(2),                    # 14
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),                    # 7
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return HUDGrayCNN()


# ── Training driver ─────────────────────────────────────────────────

def train_and_export(
    X: np.ndarray, y: np.ndarray, counts: dict,
) -> Tuple[float, dict, np.ndarray, np.ndarray]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("[train] device=%s torch=%s", device, torch.__version__)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    split = int(len(perm) * (1 - VAL_SPLIT))
    tr_idx, va_idx = perm[:split], perm[split:]

    X_tr = torch.from_numpy(X[tr_idx]).to(device)
    y_tr = torch.from_numpy(y[tr_idx]).to(device)
    X_va = torch.from_numpy(X[va_idx]).to(device)
    y_va = torch.from_numpy(y[va_idx]).to(device)

    bincnt = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    safe = np.maximum(bincnt, 1.0)
    inv = float(safe.sum()) / (float(NUM_CLASSES) * safe)
    median_w = float(np.median(inv))
    weights = np.minimum(inv, median_w * 5.0).astype(np.float32)
    log.info("[train] class weights:")
    for ch, w, c in zip(CHAR_CLASSES, weights, bincnt):
        log.info("  %r: count=%d weight=%.3f", ch, int(c), float(w))
    weights_t = torch.from_numpy(weights).to(device)

    model = build_model(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5,
    )

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=128)

    log.info(
        "[train] epochs=%d train=%d val=%d", EPOCHS, len(tr_idx), len(va_idx),
    )

    best_val = 0.0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        loss_sum, n_seen, n_correct = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(xb)
            n_correct += int((logits.argmax(1) == yb).sum().item())
            n_seen += len(xb)
        scheduler.step()
        train_loss = loss_sum / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)

        model.eval()
        v_correct, v_seen = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                v_correct += int((logits.argmax(1) == yb).sum().item())
                v_seen += len(xb)
        val_acc = v_correct / max(v_seen, 1)

        log.info(
            "  epoch %3d/%d  loss=%.4f  train=%.1f%%  val=%.1f%%",
            epoch + 1, EPOCHS, train_loss, train_acc * 100, val_acc * 100,
        )

        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No epoch improved val accuracy — refusing to save.")
    model.load_state_dict(best_state)
    model.eval().to(device)

    # ── Per-class val accuracy + confusion matrix ──
    K = NUM_CLASSES
    confusion = np.zeros((K, K), dtype=np.int64)
    per_class_correct = np.zeros(K, dtype=np.int64)
    per_class_total = np.zeros(K, dtype=np.int64)
    with torch.no_grad():
        logits = model(X_va)
        pred = logits.argmax(1).cpu().numpy()
        truth = y_va.cpu().numpy()
    for p, t in zip(pred, truth):
        confusion[int(t), int(p)] += 1
        per_class_total[int(t)] += 1
        if p == t:
            per_class_correct[int(t)] += 1

    log.info("[train] per-class val accuracy:")
    per_class_acc: dict = {}
    for i, ch in enumerate(CHAR_CLASSES):
        if per_class_total[i] == 0:
            log.info("  %r: (no val samples)", ch)
            per_class_acc[ch] = None
        else:
            acc = per_class_correct[i] / per_class_total[i]
            per_class_acc[ch] = float(acc)
            log.info(
                "  %r: %d/%d = %.1f%%",
                ch, int(per_class_correct[i]), int(per_class_total[i]),
                acc * 100,
            )

    log.info("[train] confusion matrix (rows=truth, cols=pred):")
    header = "      " + "  ".join(f"{c:>4}" for c in CHAR_CLASSES)
    log.info(header)
    for i, ch in enumerate(CHAR_CLASSES):
        row = "  ".join(f"{int(confusion[i, j]):4d}" for j in range(K))
        log.info("  %s  %s", ch, row)

    # ── Export to ONNX (opset 13, dynamic batch axis) ──
    # Force the legacy TorchScript-based exporter (dynamo=False) — the
    # new dynamo path prints unicode markers that crash on cp1252
    # consoles, and the old path is rock-solid for this opset/arch.
    import torch
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model, dummy, str(OUT_ONNX),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    log.info("[train] wrote ONNX: %s", OUT_ONNX)

    return float(best_val), per_class_acc, confusion, truth


def main() -> int:
    log.info("=== train_hud_cnn (grayscale 12-class HUD CNN) ===")
    log.info("region kind:   %s", REGION_KIND)
    log.info("label set:     %r (%d classes)", CHAR_CLASSES, NUM_CLASSES)
    log.info("output ONNX:   %s", OUT_ONNX)
    log.info("output META:   %s", OUT_META)
    log.info("output LOG:    %s", OUT_LOG)

    # HARD GUARD: resolve staging via registry, then verify the
    # resolved path is one the spec sanctions.
    staging = resolve_staging_dir(REGION_KIND)
    log.info("staging dir:   %s", staging)
    # Re-assert: the resolved dir MUST be inside the registered sources.
    # ``resolve_staging_dir`` picks from the sources list, but a future
    # refactor could drift — make the guarantee explicit.
    assert_path_belongs_to(REGION_KIND, staging)

    rng = random.Random(SEED)
    t0 = time.time()
    X, y, counts = load_dataset(staging, rng)
    log.info("[load] X shape=%s y shape=%s", X.shape, y.shape)
    if len(X) == 0:
        log.error("[!] No data loaded.")
        return 1

    best_val, per_class_acc, confusion, val_truth = train_and_export(
        X, y, counts,
    )
    dt = time.time() - t0

    val_samples_per_class = {
        ch: int(np.sum(val_truth == i))
        for i, ch in enumerate(CHAR_CLASSES)
    }

    meta = {
        "kind": REGION_KIND,
        "version": "v1",
        "charClasses": CHAR_CLASSES,
        "numClasses": NUM_CLASSES,
        "inputShape": [1, 1, IMG_SIZE, IMG_SIZE],
        "valAccuracy": best_val,
        "trainSamples": int(len(y) - sum(val_samples_per_class.values())),
        "valSamples": int(sum(val_samples_per_class.values())),
        "perClassCounts": {ch: int(counts.get(ch, 0)) for ch in CHAR_CLASSES},
        "perClassValAccuracy": per_class_acc,
        "perClassValSamples": val_samples_per_class,
        "confusionMatrix": confusion.tolist(),
        "augmentationPerSample": N_AUGS_PER_SAMPLE,
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trainingSeconds": dt,
        "modelPath": str(OUT_ONNX),
        "stagingDir": str(staging),
        "notes": (
            "HUD-specific grayscale per-glyph CNN. 12 classes "
            "(0-9, '.', '%') covering mass / resistance / "
            "instability fields. Trained ONLY on HUD-registered "
            "sources (training_data_user_panel/) — never on "
            "signature staging dirs. The HUD font differs from the "
            "signature font (smaller glyph height, different stroke "
            "weight), so the signature CNN is a poor substitute for "
            "HUD glyphs. Replaces the shipped model_cnn.onnx which "
            "has no provenance info but was trained on a similar "
            "alphabet; this one's val accuracy + confusion matrix "
            "are documented in this JSON sidecar."
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("[meta] wrote %s", OUT_META)

    log.info(
        "[done] best_val=%.2f%%  total_seconds=%.1f",
        best_val * 100, dt,
    )
    return 0 if best_val >= 0.85 else 2


if __name__ == "__main__":
    sys.exit(main())
