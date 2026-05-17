"""Trainer for the RGB per-glyph mining-HUD CNN.

RGB twin of ``train_hud_cnn.py`` — same 3-conv-block architecture,
but with a 3-channel input (the runtime preserves cyan-vs-bg colour
and chromatic-aberration patterns that the grayscale CNN collapses
to luma). 12-class output covers the full HUD alphabet
``0123456789.%``.

**HUD font ≠ Signature font.** This trainer reads ONLY from
HUD-registered RGB staging dirs (``training_data_user_panel_rgb/``)
populated by ``scripts/extract_hud_glyph_crops_rgb.py`` which itself
walks ``training_data_hud_crops/`` — both HUD-only by construction.
Every PNG path is run through ``assert_path_belongs_to("hud_rgb", ...)``
before being loaded.

Inputs:
  * Digits 0-9: ``training_data_user_panel_rgb/{0,1,...,9}/*.png``
  * Decimal point: ``training_data_user_panel_rgb/dot/*.png``
  * Percent: ``training_data_user_panel_rgb/pct/*.png``

Outputs (paths from ``REGIONS["hud_rgb"].model_path``):
  * ``ocr/models/model_hud_rgb_cnn.onnx``      — opset 13 ONNX
  * ``ocr/models/model_hud_rgb_cnn.json``      — alphabet / val metrics
  * ``ocr/models/model_hud_rgb_cnn_train.log`` — training log

Run::

    %LOCALAPPDATA%\\Python\\pythoncore-3.14-64\\python.exe \\
        ocr/train_hud_rgb_cnn.py
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


THIS_FILE = Path(__file__).resolve()
TOOL_DIR = THIS_FILE.parent.parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from ocr.training_registry import (        # noqa: E402
    get as _registry_get,
    resolve_staging_dir,
    assert_path_belongs_to,
)

REGION_KIND = "hud_rgb"
SPEC = _registry_get(REGION_KIND)

CHAR_CLASSES = SPEC.label_set              # "0123456789.%" — 12 chars
assert CHAR_CLASSES == "0123456789.%", (
    f"HUD-RGB label set drift — spec says {CHAR_CLASSES!r}, expected "
    f"'0123456789.%' (12 chars). Refusing to train."
)
NUM_CLASSES = len(CHAR_CLASSES)             # 12

FOLDER_LABEL_MAP = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "dot": ".",
    "pct": "%",
}

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

# RGB tiles are scarcer than the grayscale corpus (the extractor only
# emits when the segmenter glyph count matches the label length).
# Heavier augmentation makes up the difference.
N_AUGS_PER_SAMPLE = 8

FLOOR_PER_CLASS = SPEC.floor_per_class       # 30


log = logging.getLogger("train_hud_rgb_cnn")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(OUT_LOG, mode="w", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(message)s"))
log.handlers.clear()
log.addHandler(_fh)
log.addHandler(_sh)


# ── Augmentation utilities (RGB) ────────────────────────────────────

def _affine_jitter(
    img: Image.Image, *,
    max_rot_deg: float,
    max_trans_frac: float,
    min_scale: float,
    max_scale: float,
    rng: random.Random,
) -> Image.Image:
    """Rotate + translate + scale jitter on an RGB image.

    Fill is white (255, 255, 255) to match the grayscale-segmenter's
    pad colour — keeps augmented and natural samples within the same
    visual distribution.
    """
    angle = rng.uniform(-max_rot_deg, max_rot_deg)
    sx = rng.uniform(min_scale, max_scale)
    sy = rng.uniform(min_scale, max_scale)
    tx = rng.uniform(-max_trans_frac, max_trans_frac) * img.size[0]
    ty = rng.uniform(-max_trans_frac, max_trans_frac) * img.size[1]
    out = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    nw = max(1, int(round(out.size[0] * sx)))
    nh = max(1, int(round(out.size[1] * sy)))
    out = out.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", img.size, (255, 255, 255))
    cx = (img.size[0] - nw) // 2 + int(round(tx))
    cy = (img.size[1] - nh) // 2 + int(round(ty))
    canvas.paste(out, (cx, cy))
    return canvas


def _photo_jitter(img: Image.Image, *, rng: random.Random) -> Image.Image:
    """Brightness + contrast + slight saturation jitter."""
    b = rng.uniform(0.80, 1.20)
    c = rng.uniform(0.80, 1.20)
    s = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    img = ImageEnhance.Color(img).enhance(s)
    return img


def _augment_sample(
    src: Image.Image, *, n: int, rng: random.Random,
) -> List[Image.Image]:
    base = src.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
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
    out: dict[str, List[Path]] = {ch: [] for ch in CHAR_CLASSES}
    for folder_name, label_ch in FOLDER_LABEL_MAP.items():
        cls_dir = staging / folder_name
        if not cls_dir.is_dir():
            log.warning(
                "[load] class %r folder missing: %s", label_ch, cls_dir,
            )
            continue
        assert_path_belongs_to(REGION_KIND, cls_dir)
        for png in sorted(cls_dir.glob("*.png")):
            assert_path_belongs_to(REGION_KIND, png)
            out[label_ch].append(png)
    return out


def load_dataset(
    staging: Path, rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    by_class = _enumerate_class_files(staging)
    log.info("[load] raw per-class file counts:")
    for ch in CHAR_CLASSES:
        log.info("  %r: %d files", ch, len(by_class[ch]))
    for ch in CHAR_CLASSES:
        if len(by_class[ch]) < FLOOR_PER_CLASS:
            raise RuntimeError(
                f"Class {ch!r} has {len(by_class[ch])} samples; "
                f"floor is {FLOOR_PER_CLASS}. Run the extractor with "
                f"more captures or relax the count-mismatch filter."
            )

    images: List[np.ndarray] = []
    labels: List[int] = []
    counts: dict = {}

    for cls_idx, ch in enumerate(CHAR_CLASSES):
        n_real = 0
        n_total = 0
        for src in by_class[ch]:
            try:
                base = Image.open(src).convert("RGB")
            except Exception as exc:
                log.warning("[load] skip %s: %s", src.name, exc)
                continue
            n_real += 1
            augs = _augment_sample(base, n=N_AUGS_PER_SAMPLE, rng=rng)
            for im in augs:
                arr = np.asarray(im, dtype=np.float32) / 255.0
                # (H, W, 3) -> (3, H, W)
                images.append(arr.transpose(2, 0, 1))
                labels.append(cls_idx)
                n_total += 1
        counts[ch] = n_real
        log.info(
            "[load] class %r: %d raw → %d augmented", ch, n_real, n_total,
        )

    X = np.stack(images, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, counts


# ── Model: 3 conv blocks + FC, RGB input ────────────────────────────

def build_model(num_classes: int):
    """Architecture matching the v2 signature RGB trainer (3 conv
    blocks + FC head) — Conv2d(3, 32, 3) first layer."""
    import torch.nn as nn

    class HUDRGBCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),                  # 14
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),                  # 7
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

    return HUDRGBCNN()


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

    import torch
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
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
    log.info("=== train_hud_rgb_cnn (RGB 12-class HUD CNN) ===")
    log.info("region kind:   %s", REGION_KIND)
    log.info("label set:     %r (%d classes)", CHAR_CLASSES, NUM_CLASSES)
    log.info("output ONNX:   %s", OUT_ONNX)
    log.info("output META:   %s", OUT_META)
    log.info("output LOG:    %s", OUT_LOG)

    staging = resolve_staging_dir(REGION_KIND)
    log.info("staging dir:   %s", staging)
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
        "inputShape": [1, 3, IMG_SIZE, IMG_SIZE],
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
            "HUD-specific RGB per-glyph CNN. 12 classes "
            "(0-9, '.', '%') covering mass / resistance / "
            "instability fields. RGB twin of model_hud_cnn.onnx; "
            "preserves cyan-vs-bg colour and chromatic-aberration "
            "patterns that the grayscale CNN collapses to luma. "
            "Trained ONLY on HUD-registered sources "
            "(training_data_user_panel_rgb/ extracted from "
            "training_data_hud_crops/). Never shares training data "
            "with the signature RGB CNN — HUD font and Signature "
            "font are structurally different."
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
