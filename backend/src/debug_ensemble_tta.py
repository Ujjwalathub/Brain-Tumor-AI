"""
Ensemble / TTA analysis script.

Evaluates candidate inference-stack improvements on the held-out test set
without retraining:
  1. Single-model accuracy (EfficientNet-B0 ep19, ResNet50 FR-T10 seed42 ep21)
  2. Horizontal-flip test-time augmentation (TTA), individually + ensembled
  3. Two-model probability averaging

Only runs diagnostics; it never changes model artifacts.
Usage:
    python src/debug_ensemble_tta.py
"""

import copy
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import BrainTumorDataset
from data.transforms import get_val_transforms
from models import create_model
from utils import load_config, set_seed

MODELS = [
    (
        "efficientnet_b0",
        "configs/efficientnet_b0.yaml",
        "models/archive_efficientnet_b0/best_epoch019_macro-recall0.9929.pth",
    ),
    (
        "resnet50_seed42",
        "configs/resnet50.yaml",
        "models/fr_t10/resnet50_seed42/best_epoch021_macro-recall0.9929.pth",
    ),
]

BATCH_SIZE = 256
AMP_DTYPE = torch.float16


def load_model(config_path: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("config", config)
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(memory_format=torch.channels_last).to(device)
    return model


@torch.no_grad()
def collect_probs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    flip: bool,
    device: str,
) -> np.ndarray:
    """Softmax probabilities over test set; optionally with horizontal-flip TTA."""
    all_probs: List[np.ndarray] = []
    for images, _ in tqdm(dataloader, desc=f"TTA flip={flip}", leave=False):
        # Dataset tensors are float32 already; keep normalization, mirror pixels.
        if flip:
            images = torch.flip(images, dims=[-1])
        images = images.to(memory_format=torch.channels_last).to(device)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def metrics_summary(name: str, probs: np.ndarray, ys: np.ndarray) -> Dict:
    preds = probs.argmax(axis=1)

    def recall(cls: int) -> float:
        m = ys == cls
        return float((preds[m] == cls).mean()) if m.any() else float("nan")

    acc = float((preds == ys).mean())
    macro_recall = float(np.mean([recall(c) for c in np.unique(ys)]))
    print(f"  {name:<44} acc={acc:.4f}  macro_recall={macro_recall:.4f}  "
          f"glioma_recall={recall(0):.3f}")
    return {"name": name, "acc": acc, "macro_recall": macro_recall}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("Ensemble / TTA analysis on test split")
    print(f"Device: {device}")
    print("=" * 70)

    set_seed(42)
    workspace = Path.cwd()
    processed_dir = workspace / "data" / "processed"

    manifest = pd.read_csv(processed_dir / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].copy()
    dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=test_df,
        transform=get_val_transforms(),
    )
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    ys = np.array([int(y) for _, y in dataset])
    results: Dict[str, Dict[str, np.ndarray]] = {}

    for name, config_path, checkpoint_path in MODELS:
        ckpt = workspace / checkpoint_path
        if not ckpt.exists():
            print(f"Skipping {name}: checkpoint missing ({ckpt})")
            continue
        print(f"\nCollecting probabilities: {name}")
        model = load_model(str(workspace / config_path), str(ckpt), device)
        p_id = collect_probs(model, dataloader, flip=False, device=device)
        p_hf = collect_probs(model, dataloader, flip=True, device=device)
        results[name] = {"identity": p_id, "hflip": p_hf}
        del model
        torch.cuda.empty_cache()

    keys = list(results)
    if len(keys) < 2:
        print("Need two models for ensemble; aborting.")
        return

    a, b = keys
    pa, pb = results[a], results[b]

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    metrics_summary(f"{a} (identity)", pa["identity"], ys)
    metrics_summary(f"{b} (identity)", pb["identity"], ys)
    metrics_summary(f"{a} hflip-TTA", (pa["identity"] + pa["hflip"]) / 2, ys)
    metrics_summary(f"{b} hflip-TTA", (pb["identity"] + pb["hflip"]) / 2, ys)
    metrics_summary(
        "ensemble avg (identity)",
        (pa["identity"] + pb["identity"]) / 2, ys,
    )
    metrics_summary(
        "ensemble avg + hflip-TTA",
        (pa["identity"] + pa["hflip"] + pb["identity"] + pb["hflip"]) / 4, ys,
    )
    # Weighted simple ensemble (glioma recaller emphasis: heavier EffNet)
    metrics_summary(
        "ensemble 2:1 (effnet:resnet, identity)",
        (2 * pa["identity"] + pb["identity"]) / 3, ys,
    )


if __name__ == "__main__":
    main()
