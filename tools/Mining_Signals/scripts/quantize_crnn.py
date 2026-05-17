"""INT8-quantize the CRNN ONNX for CPU inference speed.

Takes ``ocr/models/model_crnn.onnx`` (fp32) and writes
``ocr/models/model_crnn_int8.onnx`` (quantized). Dynamic quantization
for simplicity — targets Linear + Conv + MatMul layers, which are
the bulk of our weights and run ~2-4× faster under int8 on CPU.

Dynamic quant avoids needing a calibration dataset. Accuracy loss
is typically < 1 percentage point vs the fp32 model. If we ever
want static quant (slightly better accuracy, slightly faster), we
can feed a ~100-sample calibration loader through ``CalibrationDataReader``
and use ``quantize_static`` instead.

Usage:
    python scripts/quantize_crnn.py

Then point the runtime at ``model_crnn_int8.onnx`` (set CRNN_MODEL_PATH
in sc_ocr/config.py or via env var).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "ocr" / "models"
FP32_PATH = MODELS / "model_crnn.onnx"
INT8_PATH = MODELS / "model_crnn_int8.onnx"
META_PATH = MODELS / "model_crnn.json"
INT8_META_PATH = MODELS / "model_crnn_int8.json"


def main() -> int:
    if not FP32_PATH.is_file():
        print(f"ERROR: no fp32 model at {FP32_PATH}", file=sys.stderr)
        return 1

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print(
            "ERROR: onnxruntime.quantization not available. "
            "Install with: pip install onnxruntime onnx",
            file=sys.stderr,
        )
        return 1

    fp32_size = FP32_PATH.stat().st_size
    print(f"Quantizing {FP32_PATH.name} ({fp32_size/1e6:.2f} MB fp32)...")

    t0 = time.time()
    quantize_dynamic(
        model_input=str(FP32_PATH),
        model_output=str(INT8_PATH),
        weight_type=QuantType.QUInt8,
        # Don't quantize activations — keeps LSTM accuracy stable.
        # Dynamic quant only quantizes weights; activations stay fp32
        # and are scaled per-batch at inference time.
    )
    elapsed = time.time() - t0
    int8_size = INT8_PATH.stat().st_size
    print(f"  wrote {INT8_PATH.name} ({int8_size/1e6:.2f} MB int8)")
    print(f"  compression: {fp32_size/int8_size:.2f}×")
    print(f"  elapsed: {elapsed:.1f}s")

    # Mirror the metadata JSON
    if META_PATH.is_file():
        import json
        with open(META_PATH) as f:
            meta = json.load(f)
        meta["quantization"] = "dynamic_int8"
        meta["fp32FileMB"] = round(fp32_size / 1e6, 2)
        meta["int8FileMB"] = round(int8_size / 1e6, 2)
        with open(INT8_META_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  wrote {INT8_META_PATH.name}")

    # Quick sanity load
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(
            str(INT8_PATH),
            providers=["CPUExecutionProvider"],
        )
        inp_name = sess.get_inputs()[0].name
        # Dummy forward at a realistic shape
        dummy = np.random.randn(1, 1, 32, 160).astype(np.float32)
        t1 = time.time()
        _ = sess.run(None, {inp_name: dummy})
        t_ms = (time.time() - t1) * 1000
        print(f"  sanity inference: {t_ms:.1f} ms on CPU")
    except Exception as exc:
        print(f"  WARNING: sanity check failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
