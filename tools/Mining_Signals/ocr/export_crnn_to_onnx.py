"""Export model_signal_crnn_rgb_best.pth -> model_signal_crnn_rgb.onnx.

Lets us salvage a useful ONNX model when training got killed before
its own export step (e.g. the harness reaped the background bash at
77 minutes). The training script saves a ``.pth`` checkpoint every
time val accuracy improves, so this exporter pulls the best-so-far
weights off disk and runs torch.onnx.export against them.

Also writes a metadata JSON alongside the ONNX so downstream
consumers know the alphabet + input shape.

Run: ``python ocr/export_crnn_to_onnx.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

THIS = Path(__file__).resolve()
TOOL = THIS.parent.parent
sys.path.insert(0, str(TOOL))

from ocr.train_signal_crnn_rgb import RGBCRNN, ALPHABET, BLANK_IDX, H_TARGET  # noqa: E402

MODELS_DIR = THIS.parent / "models"
CKPT_PATH = MODELS_DIR / "model_signal_crnn_rgb_best.pth"
ONNX_PATH = MODELS_DIR / "model_signal_crnn_rgb.onnx"
META_PATH = MODELS_DIR / "model_signal_crnn_rgb.json"


def main() -> int:
    if not CKPT_PATH.is_file():
        print(f"FATAL: no checkpoint at {CKPT_PATH}")
        return 1
    print(f"Loading checkpoint: {CKPT_PATH}")
    saved = torch.load(str(CKPT_PATH), map_location="cpu")
    if isinstance(saved, dict) and "state_dict" in saved:
        state_dict = saved["state_dict"]
        val_loss = float(saved.get("val_loss", float("nan")))
        val_acc = float(saved.get("val_acc", float("nan")))
        epoch = int(saved.get("epoch", -1))
        print(f"  metadata: epoch={epoch} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
    else:
        state_dict = saved
        val_loss = float("nan")
        val_acc = float("nan")
        epoch = -1
        print("  (no metadata in checkpoint -- raw state_dict)")

    model = RGBCRNN()
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # Dummy input for tracing. Width 200 px is arbitrary; the
    # dynamic_axes below let the exported ONNX accept any width.
    dummy = torch.randn(1, 3, H_TARGET, 200, dtype=torch.float32)
    print(f"Exporting ONNX to {ONNX_PATH}")
    # PyTorch 2.11's new dynamo-based ONNX exporter doesn't support
    # adaptive_max_pool2d with dynamic shapes — and the agent's
    # architecture uses that op to collapse the height axis. The
    # legacy TorchScript exporter (``dynamo=False``) handles it
    # cleanly. Explicit kwarg keeps this stable across PyTorch
    # version bumps.
    torch.onnx.export(
        model, dummy, str(ONNX_PATH),
        opset_version=14,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 3: "width"},
            "logits": {1: "batch", 0: "time"},
        },
        dynamo=False,
    )
    onnx_size = ONNX_PATH.stat().st_size
    print(f"  wrote {onnx_size} bytes")

    # Verify ONNX loads + runs cleanly.
    try:
        import onnxruntime as _ort
        sess = _ort.InferenceSession(
            str(ONNX_PATH), providers=["CPUExecutionProvider"],
        )
        out_names = [o.name for o in sess.get_outputs()]
        in_names = [i.name for i in sess.get_inputs()]
        # Run on the dummy to confirm shapes.
        result = sess.run(None, {in_names[0]: dummy.numpy()})
        print(f"  ONNX runtime verify OK: inputs={in_names} outputs={out_names} "
              f"logits.shape={result[0].shape}")
    except Exception as exc:
        print(f"  WARNING: ONNX runtime verification failed: {exc}")

    # Metadata JSON.
    meta = {
        "schema": "crnn_rgb_v1",
        "alphabet": ALPHABET,
        "blank_idx": BLANK_IDX,
        "input_shape": [None, 3, H_TARGET, None],
        "input_normalization": "rgb_u8_to_float[0,1] then polarity-canonicalized per-channel",
        "checkpoint_val_acc": val_acc,
        "checkpoint_val_loss": val_loss,
        "checkpoint_epoch": epoch,
        "source_checkpoint": CKPT_PATH.name,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {META_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
