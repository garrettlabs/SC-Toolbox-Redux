"""PyTorch CNN trainer for SC HUD digit classification.

Reads 28x28 grayscale glyphs from
    OCR_TRAIN_SOURCE/<class>/*.png  (default: training_data_user_panel/)

Trains a small CNN (2 conv + 2 FC) on GPU if available, exports to
    ocr/models/torch_digit.pt
plus a metadata JSON.

Class folders: 0..9, dot, pct (12 classes total).

Usage:
    OCR_TRAIN_SOURCE=path/to/glyphs \
      python -m ocr.train_torch [--epochs 20] [--lr 0.001]
    python -m ocr.train_torch --stats
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


_MODULE_DIR = Path(__file__).parent
_TRAINING_DIR = Path(
    os.environ.get("OCR_TRAIN_SOURCE")
    or str(_MODULE_DIR.parent / "training_data_user_panel")
)
_MODEL_DIR = _MODULE_DIR / "models"
_MODEL_PATH = _MODEL_DIR / "torch_digit.pt"
_META_PATH = _MODEL_DIR / "torch_digit.json"

CLASS_MAP = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "dot": ".", "pct": "%",
}


def load_training_data(src: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes: list[str] = []
    images: list[np.ndarray] = []
    labels: list[int] = []
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        char = CLASS_MAP.get(cls_dir.name)
        if not char:
            continue
        idx = len(classes)
        classes.append(char)
        for f in cls_dir.glob("*.png"):
            try:
                img = Image.open(f).convert("L").resize((28, 28), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                images.append(arr.reshape(1, 28, 28))
                labels.append(idx)
            except Exception as exc:
                print(f"  skip {f.name}: {exc}", flush=True)
    if not images:
        return np.array([]), np.array([]), classes
    return (
        np.stack(images).astype(np.float32),
        np.array(labels, dtype=np.int64),
        classes,
    )


def print_stats(y: np.ndarray, classes: list[str]) -> None:
    print("Dataset statistics:", flush=True)
    for i, c in enumerate(classes):
        n = int(np.sum(y == i))
        bar = "#" * min(n // 4, 50)
        print(f"  {c!r}: {n:5d} {bar}", flush=True)
    print(f"  Total: {len(y)} samples across {len(classes)} classes", flush=True)


def build_cnn(num_classes: int):
    """Larger CNN with BatchNorm — more capacity, generalizes better with augmentation."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),                    # 14x14
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),                    # 7x7
        nn.Flatten(),
        nn.Linear(128 * 7 * 7, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )


def augment_batch(xb, training: bool = True):
    """Random shift/rotation/zoom for training. CPU-side, simple PIL-free.
    Input: torch tensor (B, 1, 28, 28) on any device.
    """
    import torch
    import torch.nn.functional as F
    if not training:
        return xb
    B = xb.shape[0]
    device = xb.device
    # Random affine matrix per sample: translate ±0.1, rotate ±10°, scale 0.9-1.1
    angles = (torch.rand(B, device=device) - 0.5) * 0.35  # ±0.175 rad ≈ ±10°
    tx = (torch.rand(B, device=device) - 0.5) * 0.2
    ty = (torch.rand(B, device=device) - 0.5) * 0.2
    scales = 0.9 + torch.rand(B, device=device) * 0.2  # 0.9..1.1
    cos = torch.cos(angles) / scales
    sin = torch.sin(angles) / scales
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, xb.size(), align_corners=False)
    out = F.grid_sample(xb, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out


def train_model(epochs: int, lr: float, X: np.ndarray, y: np.ndarray, classes: list[str]) -> tuple[float, dict]:
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}", flush=True)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # 85/15 split with stratification (manual to avoid sklearn dep)
    rng = np.random.default_rng(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    val_size = max(len(X) // 7, len(classes) * 2)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    X_tr = torch.from_numpy(X[tr_idx]).to(device)
    y_tr = torch.from_numpy(y[tr_idx]).to(device)
    X_val = torch.from_numpy(X[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)

    model = build_cnn(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine LR schedule — warm-up + decay over all epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Class-weighted loss — boost rare classes (inverse frequency)
    class_counts = torch.bincount(y_tr, minlength=len(classes)).float()
    class_weights = (class_counts.sum() / (class_counts * len(classes))).clamp(0.5, 3.0)
    print(f"  class weights: {dict(zip(classes, [round(float(w),2) for w in class_weights]))}", flush=True)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val = 0.0
    best_state = None
    batch_size = 64
    print(f"  train={len(tr_idx)}  val={len(val_idx)}  epochs={epochs}  augment=ON", flush=True)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        loss_sum = 0.0
        for i in range(0, len(X_tr), batch_size):
            bi = perm[i:i + batch_size]
            xb = augment_batch(X_tr[bi], training=True)
            yb = y_tr[bi]
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(bi)
        sched.step()
        avg_loss = loss_sum / len(X_tr)

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            pred = logits.argmax(dim=1)
            val_acc = float((pred == y_val).float().mean())

        cur_lr = sched.get_last_lr()[0]
        print(f"  epoch {ep+1:3d}: loss={avg_loss:.4f}  val_acc={val_acc*100:.1f}%  lr={cur_lr:.5f}", flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best_val, model.state_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument(
        "--inverted", action="store_true",
        help=(
            "Train the polarity-INVERTED sibling classifier.  Reads "
            "training_data_user_panel_inv/ (produced by "
            "scripts/make_inverted_dataset.py) and writes "
            "models/torch_digit_inv.{pt,json}.  Then run "
            "scripts/export_torch_to_onnx.py --inverted to deploy as "
            "model_cnn_inv.onnx."
        ),
    )
    args = parser.parse_args()

    # Pick source + output paths based on --inverted.  Don't bind these
    # until after parsing args so the --inverted flag can override the
    # module-level defaults set at import time.
    if args.inverted:
        # Inverted pipeline: read inverted user_panel, write _inv
        # artifacts so we don't clobber the canonical torch_digit.{pt,json}.
        src_dir = _MODULE_DIR.parent / "training_data_user_panel_inv"
        model_path = _MODEL_DIR / "torch_digit_inv.pt"
        meta_path = _MODEL_DIR / "torch_digit_inv.json"
    else:
        src_dir = _TRAINING_DIR
        model_path = _MODEL_PATH
        meta_path = _META_PATH

    print(f"Loading training data from {src_dir}...", flush=True)
    X, y, classes = load_training_data(src_dir)
    if len(X) == 0:
        print("No training data found.", flush=True)
        return
    print_stats(y, classes)
    if args.stats:
        return
    if len(X) < 60:
        print(f"Only {len(X)} samples — need at least 60 for CNN.", flush=True)
        return

    t0 = time.time()
    best_acc, state = train_model(args.epochs, args.lr, X, y, classes)
    dt = time.time() - t0

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save({
        "state_dict": state,
        "classes": classes,
        "input_shape": [1, 28, 28],
    }, model_path)

    meta = {
        "classes": classes,
        "samples": len(X),
        "val_accuracy": best_acc,
        "model_type": "CNN (Conv2d-32, Conv2d-64, FC-128)",
        "training_seconds": dt,
        "epochs": args.epochs,
        "lr": args.lr,
        "inverted": args.inverted,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to {model_path}", flush=True)
    print(f"Metadata:     {meta_path}", flush=True)
    print(f"Best validation accuracy: {best_acc*100:.1f}%", flush=True)
    if args.inverted:
        print()
        print("Next: export to ONNX for runtime use:")
        print("    python scripts/export_torch_to_onnx.py --inverted", flush=True)


if __name__ == "__main__":
    main()
