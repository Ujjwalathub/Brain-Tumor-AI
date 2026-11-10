"""Probe: ORT CUDA EP with cudnn from torch/lib (PATH + add_dll_directory)."""

import os
import sys
from pathlib import Path

import importlib.util

spec = importlib.util.find_spec("torch")
torch_lib = Path(spec.origin).parent / "lib"
ort_capi = Path(importlib.util.find_spec("onnxruntime").origin).parent / "capi"

os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ["PATH"]
os.add_dll_directory(str(torch_lib))
os.add_dll_directory(str(ort_capi))

# Where does cudnn64_9.dll actually resolve from?
import ctypes
h = ctypes.WinDLL("cudnn64_9.dll")
sym = getattr(h, "cudnnCreate", None)
print("cudnnCreate symbol:", "FOUND" if sym else "MISSING")
print("torch/lib exists:", torch_lib.is_dir())
print("ort capi exists:", ort_capi.is_dir())

import onnxruntime as ort

sess_opts = ort.SessionOptions()
sess_opts.log_severity_level = 3
sess = ort.InferenceSession(
    "models/efficientnet_b0.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("active provider:", sess.get_providers()[0])
data = sess.run(None, {"input": __import__("numpy").zeros((1, 3, 224, 224), dtype=__import__("numpy").float32)})
print("run OK, output shape:", data[0].shape)
