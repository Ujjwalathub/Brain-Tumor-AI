"""Real-image CPU vs GPU forward comparison using the fixed seeded input."""

import sys

sys.path.insert(0, "src")

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from utils import load_config
from data.dataset import CLASS_TO_IDX
from data.transforms import get_val_transforms
from models import create_model

CKPT = "models/best_epoch019_macro-recall0.9929.pth"
IDX = {v: k for k, v in CLASS_TO_IDX.items()}


def make_models(config, sd):
    """Build independent CPU and CUDA model copies from the same weights."""
    models = {"cpu": create_model(config)}
    models["cpu"].load_state_dict({k: v.contiguous(memory_format=torch.contiguous_format) for k, v in sd.items()})
    if torch.cuda.is_available():
        m = create_model(config)
        m.load_state_dict(sd)
        models["cuda"] = m.cuda()
    return models


def run(models, device_name, tensor, amp):
    model = models[device_name].eval()
    if device_name == "cuda":
        tensor = tensor.cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(tensor).float().cpu().numpy()
    else:
        with torch.no_grad():
            out = model(tensor).numpy()
    return out


def main():
    config = load_config("configs/efficientnet_b0.yaml")
    processed = __import__("pathlib").Path.cwd() / config["data"]["processed_dir"]
    manifest = pd.read_csv(processed / "manifest.csv")
    df = manifest[manifest["split"] == "val"].head(24)

    transform = get_val_transforms(config)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    models = make_models(ckpt.get("config", config), ckpt["model_state_dict"])

    m_ort_cpu = ort.InferenceSession(
        "models/efficientnet_b0.onnx", providers=["CPUExecutionProvider"]
    )

    rows = []
    for filepath, cls in zip(df["filepath"], df["class"]):
        proc = transform(np.load(processed / filepath))
        batch = proc.unsqueeze(0)
        preds = {}
        preds["torch_cpu_fp32"] = run(models, "cpu", batch, False).flatten()

        if "cuda" in models:
            preds["torch_gpu_fp16"] = run(models, "cuda", batch, True).flatten()
            preds["torch_gpu_fp32"] = run(models, "cuda", batch, False).flatten()
        preds["ort_cpu"] = m_ort_cpu.run(None, {"input": batch.numpy()})[0].flatten()

        argmaxes = {k: int(np.argmax(v)) for k, v in preds.items()}
        base = preds["torch_cpu_fp32"]
        gpu_key = "torch_gpu_fp32" if "torch_gpu_fp32" in preds else "torch_cpu_fp32"
        rows.append({
            "file": filepath,
            "true": cls,
            "arg_cpu": IDX[argmaxes["torch_cpu_fp32"]],
            "arg_gpu": IDX[argmaxes["torch_gpu_fp16"]] if "torch_gpu_fp16" in argmaxes else "-",
            "arg_ort": IDX[argmaxes["ort_cpu"]],
            "agreement": all(a == argmaxes["torch_gpu_fp32"] for a in argmaxes.values()) if "torch_gpu_fp32" in argmaxes else False,
            "cpu_minus_gpu_max": float(np.abs(base - preds[gpu_key]).max()),
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
