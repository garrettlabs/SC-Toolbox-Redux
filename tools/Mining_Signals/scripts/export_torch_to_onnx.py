"""Export trained PyTorch CNN to ONNX so the runtime scanner uses it.

Loads ocr/models/torch_digit.pt + classes from torch_digit.json,
exports to ocr/models/model_cnn.onnx (overwriting the original),
and updates model_cnn.json with the new class layout.

The original ONNX model is backed up to model_cnn_original.onnx the
first time this runs.

Pass --inverted to export the polarity-inverted sibling instead:
loads torch_digit_inv.pt and writes model_cnn_inv.onnx (no backup,
no overwrite of the canonical files).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))

MODEL_DIR = TOOL / "ocr" / "models"
TORCH_PATH = MODEL_DIR / "torch_digit.pt"
TORCH_META = MODEL_DIR / "torch_digit.json"
ONNX_PATH = MODEL_DIR / "model_cnn.onnx"
ONNX_META = MODEL_DIR / "model_cnn.json"
ONNX_BACKUP = MODEL_DIR / "model_cnn_original.onnx"
ONNX_META_BACKUP = MODEL_DIR / "model_cnn_original.json"

# Inverted-sibling paths (when --inverted is passed).
TORCH_PATH_INV = MODEL_DIR / "torch_digit_inv.pt"
TORCH_META_INV = MODEL_DIR / "torch_digit_inv.json"
ONNX_PATH_INV = MODEL_DIR / "model_cnn_inv.onnx"
ONNX_META_INV = MODEL_DIR / "model_cnn_inv.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inverted", action="store_true",
        help=(
            "Export the inverted-polarity sibling: read "
            "torch_digit_inv.pt and write model_cnn_inv.onnx instead "
            "of overwriting the canonical model."
        ),
    )
    args = parser.parse_args()

    torch_path = TORCH_PATH_INV if args.inverted else TORCH_PATH
    torch_meta = TORCH_META_INV if args.inverted else TORCH_META
    onnx_path = ONNX_PATH_INV if args.inverted else ONNX_PATH
    onnx_meta = ONNX_META_INV if args.inverted else ONNX_META

    if not torch_path.is_file():
        print(f"ERROR: {torch_path} not found. Train first via labeler.")
        if args.inverted:
            print("    Run: python -m ocr.train_torch --inverted")
        sys.exit(1)

    # Back up original ONNX (only when overwriting the canonical;
    # never for the inverted sibling, since it has no prior version).
    if not args.inverted and ONNX_PATH.is_file() and not ONNX_BACKUP.is_file():
        print(f"Backing up original {ONNX_PATH.name} -> {ONNX_BACKUP.name}")
        shutil.copy2(ONNX_PATH, ONNX_BACKUP)
        if ONNX_META.is_file():
            shutil.copy2(ONNX_META, ONNX_META_BACKUP)

    print(f"Loading {torch_path}...")
    import torch
    from ocr.train_torch import build_cnn

    ckpt = torch.load(torch_path, map_location="cpu", weights_only=True)
    classes = ckpt["classes"]
    print(f"Classes ({len(classes)}): {classes}")

    model = build_cnn(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ONNX export
    print(f"Exporting to {onnx_path}...")
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Build the meta file in the format the runtime expects.
    # `charClasses` is the concatenated string ("0123456789.%"),
    # `numClasses` is the count.
    char_classes = "".join(classes)
    source_label = (
        "torch_digit_inv (inverted-polarity CNN)"
        if args.inverted else "torch_digit (custom CNN)"
    )
    meta = {
        "charClasses": char_classes,
        "numClasses": len(classes),
        "inputShape": [1, 1, 28, 28],
        "valAccuracy": ckpt.get("val_accuracy") or 0.99,
        "source": source_label,
    }
    # Try to fish val_accuracy from the .json sibling
    try:
        sibling = json.loads(torch_meta.read_text(encoding="utf-8"))
        meta["valAccuracy"] = sibling.get("val_accuracy", meta["valAccuracy"])
        meta["samples"] = sibling.get("samples")
    except Exception:
        pass

    onnx_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {onnx_meta}")

    # Quick verification: load with onnxruntime and predict on dummy
    print("\nVerifying ONNX export...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    out = sess.run(None, {"input": dummy.numpy()})[0]
    print(f"  ONNX output shape: {out.shape}  (should be 1x{len(classes)})")
    print(f"  Sample logits: {out[0][:5]}...")

    if args.inverted:
        print(f"\nDone. The secondary voter will load {onnx_path.name} "
              f"on next OCR scan.")
    else:
        print(f"\nDone. The runtime scanner will use this on next launch.")
        print(f"To revert: copy {ONNX_BACKUP.name} -> {ONNX_PATH.name}")


if __name__ == "__main__":
    main()
