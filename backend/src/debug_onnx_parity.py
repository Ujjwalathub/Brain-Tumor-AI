"""
ONNX parity debug - pinpoint the stage-11 parity failure.

Checks, on a fixed seeded input:
  1. Initializer census: does the ONNX graph contain ~4,012,672 params?
  2. Torch CPU logits vs ORT CPU logits (same input).
  3. Torch CUDA logits vs ORT CUDA logits.
  4. Per-output margin between argmax classes.
"""

from pathlib import Path

import numpy as np
import onnx
import torch
import onnxruntime as ort

from models import create_model
from utils import load_config

CHECKPOINT = "models/best_epoch019_macro-recall0.9929.pth"
ONNX_PATH = "models/efficientnet_b0.onnx"
CONFIG = "configs/efficientnet_b0.yaml"


def main() -> None:
    torch.manual_seed(0)
    dummy = torch.randn(1, 3, 224, 224)

    # --- 1. Initializer census ----------------------------------------
    model_proto = onnx.load(ONNX_PATH)
    init_elems = sum(np.prod(list(i.dims)) for i in model_proto.graph.initializer)
    print(f"Graph initializers: {len(model_proto.graph.initializer)} tensors")
    print(f"Initializer element count: {init_elems:,} (expected ~4,012,672)")
    print(f"Graph opset: {[op.version for op in model_proto.opset_import]}")

    # --- 2. Torch CPU ---------------------------------------------------
    config = load_config(CONFIG)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = create_model(ckpt.get("config", config))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        torch_logits = model(dummy)
    print(f"\nTorch CPU logits: {torch_logits.numpy().round(3)}")

    # --- 3. ORT CPU ------------------------------------------------------
    session_cpu = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_cpu = session_cpu.run(None, {"input": dummy.numpy()})[0]
    print(f"ORT  CPU logits: {onnx_cpu.round(3)}")
    print(f"CPU diff: max {np.abs(torch_logits.numpy() - onnx_cpu).max():.4f}")

    # --- 4. Torch CUDA vs ORT CUDA --------------------------------------
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        with torch.no_grad():
            torch_cuda = model_cuda(dummy.cuda()).cpu().numpy()
        session_gpu = ort.InferenceSession(
            str(ONNX_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print(f"ONNX CUDA active provider: {session_gpu.get_providers()[0]}")
        onnx_gpu = session_gpu.run(None, {"input": dummy.numpy()})[0]
        print(f"Torch CUDA logits: {torch_cuda.round(3)}")
        print(f"ORT  CUDA logits: {onnx_gpu.round(3)}")
        print(f"CUDA diff torch-vs-ort: max {np.abs(torch_cuda - onnx_gpu).max():.4f}")
        print(f"CUDA diff cpu-vs-cuda(ort): max {np.abs(onnx_cpu - onnx_gpu).max():.4f}")


if __name__ == "__main__":
    main()
