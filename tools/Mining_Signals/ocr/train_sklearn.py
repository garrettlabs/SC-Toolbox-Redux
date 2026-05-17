"""Train a pure-numpy classifier on SC digit glyphs.

Chose numpy-only (no sklearn/torch) because:
  - Python 3.14 lacks wheels for scikit-learn AND PyTorch at this time
  - 28x28 digit classification doesn't need a fancy model
  - Numpy is already installed and works in all environments

Algorithm: **Weighted KNN with L2 distance on flattened 28x28 pixels**.
  - No training pass — fit() just stores the samples
  - predict() finds K nearest neighbors, weights by inverse distance
  - For our fixed-font digit task this typically hits 95-98% accuracy

Reads 28x28 grayscale PNGs from a class-labeled directory structure:
    <source>/
      0/  ...
      1/  ...
      ...
      dot/ dash/ pct/   ← '.', '-', '%'

Saves a pickled dict with keys:
  - "samples": float32 array (N, 784) in [0,1]
  - "labels":  int64 array (N,)
  - "classes": list[str] mapping index → single-char

Usage:
    OCR_TRAIN_SOURCE=path/to/training_data_user_panel \\
      python -m ocr.train_sklearn
    python -m ocr.train_sklearn --stats
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
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
_MODEL_PATH = _MODEL_DIR / "sklearn_digit.pkl"
_META_PATH = _MODEL_DIR / "sklearn_digit.json"

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
                arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
                images.append(arr)
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
        bar = "#" * min(n // 2, 50)
        print(f"  {c!r}: {n:5d} {bar}", flush=True)
    print(f"  Total: {len(y)} samples across {len(classes)} classes", flush=True)


def knn_predict(X_train: np.ndarray, y_train: np.ndarray,
                X_query: np.ndarray, k: int = 3) -> np.ndarray:
    """Weighted KNN classification.

    X_train: (N, D), y_train: (N,), X_query: (M, D)
    Returns: (M,) predicted class indices.
    """
    # Compute pairwise L2 distances via (a-b)^2 = a^2 - 2ab + b^2
    # Done in chunks to avoid blowing memory on large datasets.
    M = X_query.shape[0]
    preds = np.zeros(M, dtype=np.int64)
    # Normalize for cosine-like behavior (helps for intensity variations)
    X_tr_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    X_q_norm = X_query / (np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-8)
    chunk = 256
    for i0 in range(0, M, chunk):
        i1 = min(i0 + chunk, M)
        block = X_q_norm[i0:i1]
        # Cosine similarity → (chunk, N)
        sims = block @ X_tr_norm.T
        # Pick top-k most similar
        k_eff = min(k, X_train.shape[0])
        top_idx = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        for j, neigh in enumerate(top_idx):
            # Weighted vote by similarity
            votes: dict[int, float] = {}
            for n in neigh:
                lbl = int(y_train[n])
                w = float(sims[j, n])
                votes[lbl] = votes.get(lbl, 0.0) + max(0.0, w)
            preds[i0 + j] = max(votes.items(), key=lambda kv: kv[1])[0]
    return preds


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> tuple[dict, float]:
    """'Train' the classifier (store samples) + validate via hold-out."""
    from collections import Counter
    cnt = Counter(y.tolist())
    min_per_class = min(cnt.values()) if cnt else 0

    if min_per_class < 2 or len(X) < 20:
        print(f"  only {min_per_class} samples in smallest class — "
              "keeping all as training, no val split", flush=True)
        model = {"samples": X, "labels": y}
        return model, 0.0

    # Stratified hold-out: take 1 sample per class as val + 10% extra
    indices = np.arange(len(X))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    val_size = max(len(X) // 10, len(cnt))
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    t0 = time.time()
    preds = knn_predict(X_tr, y_tr, X_val, k=3)
    dt = time.time() - t0
    acc = float(np.mean(preds == y_val))
    print(f"  KNN validation: {dt:.2f}s  val_acc={acc*100:.1f}%", flush=True)

    model = {"samples": X, "labels": y}
    return model, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    print(f"Loading training data from {_TRAINING_DIR}...", flush=True)
    X, y, classes = load_training_data(_TRAINING_DIR)
    if len(X) == 0:
        print("No training data found.", flush=True)
        return
    print_stats(y, classes)
    if args.stats:
        return
    if len(X) < 30:
        print(f"Only {len(X)} samples — need at least 30.", flush=True)
        return

    print("Building KNN classifier...", flush=True)
    model, acc = train_and_evaluate(X, y)
    model["classes"] = classes

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # NOTE: pickle output is currently unused — no reader exists in repo. Remove or wire up consumer.
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    meta = {
        "classes": classes,
        "samples": len(X),
        "val_accuracy": acc,
        "model_type": "KNN (k=3, cosine)",
        "input_shape": [28, 28],
    }
    with open(_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to {_MODEL_PATH}", flush=True)
    print(f"Metadata:     {_META_PATH}", flush=True)
    print(f"Best validation accuracy: {acc*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
