"""Per-region trainer driven by the training registry.

Pipeline (one-shot):
    1. Resolve raw capture sources for KIND from training_registry.
    2. Run the existing labeled-glyph extractor on those captures,
       writing 28×28 staging crops to spec.glyph_staging_dir.
       (This is destructive — old crops in the staging dir for
       characters we re-extracted are NOT removed; we ADD to the pool.
       Pass --reset to wipe the staging dir first.)
    3. Audit the staging dir against the spec's per-class thresholds.
       Refuse to train if any class is below `floor_per_class` unless
       --force is passed.
    4. Train a small CNN (same architecture as the legacy
       train_model.py — 2-conv + FC) on the staging crops.
    5. Export ONNX to spec.model_path; write metadata sidecar JSON.

Strict isolation tripwires:
    - Every raw capture path is checked against
      ``training_registry.assert_path_belongs_to(KIND, path)`` before
      the extractor sees it.
    - The staging dir is asserted to be ``spec.glyph_staging_dir``.
    - The output ONNX path is ``spec.model_path``. Cross-pollution
      is structurally impossible.

Usage:
    # Train the signal scanner classifier
    python scripts/train_for_region.py signal

    # Train the HUD classifier with augmentation + 30 epochs
    python scripts/train_for_region.py hud --epochs 30

    # Wipe staging and re-extract from scratch
    python scripts/train_for_region.py signal --reset

    # Force training even if some classes are below floor (debugging)
    python scripts/train_for_region.py signal --force

    # Skip extraction (use whatever's already in staging dir)
    python scripts/train_for_region.py signal --no-extract
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))
sys.path.insert(0, str(TOOL / "scripts"))

from ocr import training_registry  # noqa: E402

# extract_labeled_glyphs is the existing single-source extractor we
# reuse so this trainer doesn't duplicate the segmentation logic.
import extract_labeled_glyphs as _xlg  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Stage 1 — extract per-character 28×28 crops from raw captures
# ─────────────────────────────────────────────────────────────

def extract_from_registry(
    kind: str, *, reset: bool, left_mask_pct: float | None = None,
) -> dict[str, int]:
    """Walk every raw-capture dir registered for ``kind``, extract
    glyphs, write to ``spec.glyph_staging_dir/<char>/<file>.png``.
    Returns a per-character count of glyphs added this run.

    ``left_mask_pct`` is a per-region knob that blanks the leftmost
    fraction of every capture before segmentation. Used by the
    signal extractor to chop the location-pin icon off the front
    of every signature panel. ``None`` defers to the extractor's
    default."""
    spec = training_registry.get(kind)
    out_root = spec.glyph_staging_dir
    if reset:
        if out_root.is_dir():
            print(f"[reset] wiping {out_root}")
            shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Pick the per-region extractor function. extract_labeled_glyphs
    # already implements both region1 (HUD) and region2 (signal).
    if kind == "signal":
        def _call(img_path, label, out_root):
            kw = {}
            if left_mask_pct is not None:
                kw["left_mask_pct"] = left_mask_pct
            return _xlg.extract_region2_glyphs(img_path, label, out_root, **kw)
        extractor = _call
    elif kind == "hud":
        extractor = _xlg.extract_region1_glyphs
    else:
        raise NotImplementedError(
            f"No extractor wired up for kind {kind!r} yet — add a branch here."
        )

    counts: dict[str, int] = {}
    panels = 0
    sources = training_registry.get_training_sources(kind)
    if not sources:
        print(f"[!] No registered training sources for kind {kind!r}.")
        return counts

    for src_dir in sources:
        for img_path in sorted(src_dir.glob(spec.capture_image_glob)):
            # Tripwire: every input MUST belong to this kind. This is
            # belt-and-suspenders since we just iterated the registry's
            # own dirs, but a future refactor that bypasses the
            # iteration will hit this check.
            try:
                training_registry.assert_path_belongs_to(kind, img_path)
            except training_registry.RegistryError as e:
                print(f"[skip-tripwire] {img_path}: {e}")
                continue

            json_path = img_path.with_suffix(".json")
            if not json_path.is_file():
                continue
            try:
                label = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[skip-badjson] {img_path}: {e}")
                continue

            try:
                added = extractor(img_path, label, out_root)
            except Exception as e:
                print(f"[skip-extract] {img_path}: {e}")
                continue

            if added:
                panels += 1
                for ch, n in added.items():
                    counts[ch] = counts.get(ch, 0) + n

    if left_mask_pct is not None:
        print(f"[extract] left_mask_pct={left_mask_pct:.2f}")
    print(f"[extract] {sum(counts.values())} glyphs from {panels} panels "
          f"-> {out_root}")
    for ch in sorted(counts):
        print(f"  {ch!r}: +{counts[ch]}")
    return counts


# ─────────────────────────────────────────────────────────────
# Stage 2 — audit the staging dir against quality thresholds
# ─────────────────────────────────────────────────────────────

def audit_staging(kind: str) -> tuple[dict[str, int], list[str]]:
    """Count current per-class samples in the staging dir; return
    (counts, weak_classes_below_floor)."""
    spec = training_registry.get(kind)
    counts: dict[str, int] = {}
    weak: list[str] = []
    for ch in spec.label_set:
        # Filesystem-safe directory name — '.' and '%' get spelled out
        # by the existing extractor.
        cls_dir_name = _class_to_dirname(ch)
        cls_dir = spec.glyph_staging_dir / cls_dir_name
        n = 0
        if cls_dir.is_dir():
            n = sum(1 for p in cls_dir.glob("*.png"))
        counts[ch] = n
        if n < spec.floor_per_class:
            weak.append(ch)
    return counts, weak


def _class_to_dirname(ch: str) -> str:
    """The existing extractor encodes '.' as 'dot' and '%' as 'pct'
    on disk because '.' on a folder name is allowed but ugly and
    hidden-file-ish on some systems. '@' is the icon class for the
    signal model — its training samples live in an ``icon/`` folder
    augmented from ``training_data_blacklist/`` PNGs."""
    if ch == ".":
        return "dot"
    if ch == "%":
        return "pct"
    if ch == ",":
        return "comma"
    if ch == "@":
        return "icon"
    return ch


# ─────────────────────────────────────────────────────────────
# Stage 3 — train CNN on staging crops, export ONNX to model_path
# ─────────────────────────────────────────────────────────────

def _load_dataset(kind: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Load every PNG in the staging dir as (C, 28, 28) float32 in
    [0, 1] alongside its class index. Class order follows
    spec.label_set, which becomes the "charClasses" string in the
    exported ONNX metadata.

    Channel count is determined by ``kind``:
      * ``_rgb`` suffix: load as RGB (3 channels), shape (3, 28, 28)
      * default: load as luma (1 channel), shape (1, 28, 28)

    Polarity-inverted training: when ``kind`` ends with ``_inv``,
    each sample's pixel values are inverted (``1.0 - x``) at load
    time. The trainer uses the SAME staging directory as the non-
    inverted twin (``signal_inv`` reuses ``signal``'s pool), but the
    resulting ONNX model expects opposite-polarity inputs at runtime.
    Pairs with ``_classify_crops_signal_inv`` in the live pipeline.
    """
    spec = training_registry.get(kind)
    # Detection rules (independent — kinds can be both RGB AND inv,
    # e.g. ``signal_rgb_inv`` is the polarity-inverted twin of the RGB
    # CNN). ``endswith("_inv")`` would miss the RGB-inv combo because
    # those kinds end with ``_inv`` but contain ``_rgb`` as a middle
    # token; the more permissive ``in`` check catches both.
    parts = kind.split("_")
    invert_polarity = "inv" in parts
    is_rgb = "rgb" in parts
    n_channels = 3 if is_rgb else 1
    pil_mode = "RGB" if is_rgb else "L"
    images: list[np.ndarray] = []
    labels: list[int] = []
    char_classes = spec.label_set
    for cls_idx, ch in enumerate(char_classes):
        cls_dir = spec.glyph_staging_dir / _class_to_dirname(ch)
        if not cls_dir.is_dir():
            continue
        for png in cls_dir.glob("*.png"):
            try:
                arr = np.asarray(
                    Image.open(png).convert(pil_mode), dtype=np.float32,
                )
            except Exception:
                continue
            expected_shape = (28, 28, 3) if is_rgb else (28, 28)
            if arr.shape != expected_shape:
                try:
                    pil = Image.fromarray(arr.astype(np.uint8), mode=pil_mode)
                    pil = pil.resize((28, 28), Image.LANCZOS)
                    arr = np.asarray(pil, dtype=np.float32)
                except Exception:
                    continue
            normalized = arr / 255.0
            if invert_polarity:
                normalized = 1.0 - normalized
            images.append(normalized)
            labels.append(cls_idx)
    if not images:
        return (np.empty((0, n_channels, 28, 28), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                char_classes)
    if is_rgb:
        # Stack as (N, H, W, 3) then transpose to (N, 3, H, W).
        X = np.stack(images, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
    else:
        # (N, H, W) → (N, 1, H, W).
        X = np.stack(images, axis=0)[:, None, :, :]
    y = np.asarray(labels, dtype=np.int64)
    if invert_polarity:
        print(f"[load] polarity inversion ON — kind={kind!r} ends with '_inv'")
    if is_rgb:
        print(f"[load] RGB MODE ON — kind={kind!r} ends with '_rgb', channels=3")
    return X, y, char_classes


def _build_model(num_classes: int, in_channels: int = 1):
    """Same architecture as the legacy ocr/train_model.py for
    runtime ABI compatibility — 2-conv + dense classifier.

    ``in_channels`` controls the first conv layer's input depth.
    Default 1 (grayscale CNN, ABI-compatible with legacy models);
    pass 3 for RGB inputs.
    """
    import torch.nn as nn

    class DigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
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

    return DigitCNN()


def train_and_export(
    kind: str,
    *,
    epochs: int = 25,
    lr: float = 1e-3,
    val_split: float = 0.15,
    seed: int = 1337,
) -> Optional[float]:
    """Train and export. Returns best val accuracy in [0, 1] on
    success, None if training was skipped (no data)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    spec = training_registry.get(kind)
    print(f"[train] kind={kind!r} model_out={spec.model_path}")

    X, y, char_classes = _load_dataset(kind)
    if len(X) == 0:
        print("[!] Staging dir is empty — extract glyphs first.")
        return None

    print(f"[train] dataset: {len(X)} crops across {len(char_classes)} classes")
    bincounts = np.bincount(y, minlength=len(char_classes))
    for cls_idx, ch in enumerate(char_classes):
        print(f"        {ch!r}: {bincounts[cls_idx]}")

    if len(X) < 50:
        print("[!] Need at least 50 total samples to train meaningfully.")
        return None

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    split = int(len(perm) * (1 - val_split))
    train_idx, val_idx = perm[:split], perm[split:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=64,
    )

    # ── Class-balanced loss ──
    # Without weighting, training is dominated by the largest class.
    # On the signal pool, ``0`` has 2803 samples while ``8`` has 207
    # (~14× fewer); the unweighted loss steers the model's prior
    # toward "0" and similar round shapes, so a borderline ``8`` at
    # inference time gets confidently misread as ``6`` / ``0``. We
    # use INVERSE-FREQUENCY weights — every class's loss contribution
    # is scaled so each class contributes equally regardless of its
    # sample count. Cap each weight at the median ratio × 5 so a
    # class with literally a single sample doesn't drown out the
    # others.
    class_counts = bincounts.astype(np.float32)
    safe_counts = np.maximum(class_counts, 1.0)
    inv_freq = float(safe_counts.sum()) / (
        float(len(char_classes)) * safe_counts
    )
    median_w = float(np.median(inv_freq))
    weights = np.minimum(inv_freq, median_w * 5.0).astype(np.float32)
    print("[train] class weights (inverse-frequency, capped at 5× median):")
    for cls_idx, ch in enumerate(char_classes):
        print(
            f"        {ch!r}: count={int(class_counts[cls_idx]):4d}  "
            f"weight={weights[cls_idx]:.3f}"
        )
    weights_t = torch.from_numpy(weights)

    # Determine input channels: RGB kinds get 3, everything else 1.
    _in_channels = X.shape[1]  # X has shape (N, C, H, W)
    model = _build_model(
        num_classes=len(char_classes), in_channels=_in_channels,
    )
    criterion = nn.CrossEntropyLoss(weight=weights_t)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"[train] {epochs} epochs  train={len(train_idx)}  val={len(val_idx)}")
    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
            train_correct += (out.argmax(1) == y_batch).sum().item()
            train_total += len(X_batch)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                out = model(X_batch)
                val_correct += (out.argmax(1) == y_batch).sum().item()
                val_total += len(X_batch)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = train_loss / max(train_total, 1)
        print(f"  epoch {epoch+1:3d}/{epochs}: "
              f"loss={avg_loss:.4f}  train={train_acc*100:.1f}%  "
              f"val={val_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        scheduler.step()

    if best_state is None:
        print("[!] No improvement during training — refusing to write model.")
        return None

    # Reload best weights and export ONNX to the registry-declared path.
    model.load_state_dict(best_state)
    model.eval()
    spec.model_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, _in_channels, 28, 28)
    torch.onnx.export(
        model, dummy, str(spec.model_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    meta = {
        "kind": kind,
        "charClasses": char_classes,
        "numClasses": len(char_classes),
        "inputShape": [1, 1, 28, 28],
        "valAccuracy": best_val_acc,
        "trainSamples": int(len(train_idx)),
        "valSamples": int(len(val_idx)),
        "perClassCounts": {
            ch: int(bincounts[i]) for i, ch in enumerate(char_classes)
        },
        "trainedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "modelPath": str(spec.model_path),
        "stagingDir": str(spec.glyph_staging_dir),
    }
    meta_path = spec.model_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] best val_acc={best_val_acc*100:.2f}%  "
          f"wrote {spec.model_path.name} + {meta_path.name}")
    return best_val_acc


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=training_registry.list_kinds(),
                   help="Region kind to train (signal | hud | …).")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-extract", action="store_true",
                   help="Skip the extraction pass; use existing staging contents.")
    p.add_argument("--reset", action="store_true",
                   help="Wipe the staging dir before extracting (start clean).")
    p.add_argument("--force", action="store_true",
                   help="Train even if some classes are below floor.")
    p.add_argument("--audit-only", action="store_true",
                   help="Print the staging-dir audit and exit (no extract, no train).")
    p.add_argument("--left-mask", type=float, default=None,
                   help="(signal only) Fraction of image width to blank "
                        "from the left BEFORE segmentation, to chop off the "
                        "location-pin icon. 0.30 default; raise if icons "
                        "still leak through, lower if leading digits get "
                        "clipped.")
    args = p.parse_args()

    spec = training_registry.get(args.kind)
    print(f"=== Training pipeline for kind={args.kind!r} ===")
    print(f"    raw sources:   {len(training_registry.get_training_sources(args.kind))}")
    print(f"    staging dir:   {spec.glyph_staging_dir}")
    print(f"    model output:  {spec.model_path}")
    print(f"    char classes:  {spec.label_set!r}")
    print()

    if not args.no_extract and not args.audit_only:
        extract_from_registry(
            args.kind, reset=args.reset, left_mask_pct=args.left_mask,
        )
        print()

    counts, weak = audit_staging(args.kind)
    print(f"=== Staging audit ({args.kind}) ===")
    for ch, n in counts.items():
        tier = (
            "solid" if n >= spec.solid_per_class else
            "working" if n >= spec.working_per_class else
            "marginal" if n >= spec.floor_per_class else
            "BELOW FLOOR"
        )
        print(f"    {ch!r}: {n:5d}  [{tier}]")

    if args.audit_only:
        return 0

    if weak and not args.force:
        print()
        print(f"[!] Refusing to train: classes below floor "
              f"({spec.floor_per_class}): {weak}")
        print("    Re-run with --force to override (model will overfit those classes).")
        return 2

    print()
    val_acc = train_and_export(
        args.kind,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
    )
    return 0 if val_acc is not None else 1


if __name__ == "__main__":
    sys.exit(main())
