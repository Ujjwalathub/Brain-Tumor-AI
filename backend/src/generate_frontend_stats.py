"""
Reporting utility (one-off post-deployment):
Re-runs test-split inference on an ALREADY-CHOSEN champion and writes the
per-sample statistics needed to refresh the frontend dashboard constants in
frontend/src/lib/metrics.ts (confidence histograms, micro-average ROC).

NOTE: evaluate.py is the one-shot gate used for model selection; this script
only reports statistics for a checkpoint that was already chosen on the test
set, so it does not participate in any selection decision.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import BrainTumorDataset, IDX_TO_CLASS
from data.transforms import get_val_transforms
from models import create_model
from utils import load_config, set_seed

CHECKPOINT = "models/run_augaffine/best_epoch013_macro-recall0.9988.pth"
CONFIG = "configs/efficientnet_b0_augaffine.yaml"
SPLIT = "test"
OUT_JSON = "reports/frontend_stats_ep13_augaffine.json"
OUT_CSV = "reports/test_sample_predictions_ep13_augaffine.csv"
N_ROC_POINTS = 12


@torch.no_grad()
def main():
    config = load_config(CONFIG)
    set_seed(config.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed_dir = Path(config["data"]["processed_dir"])
    manifest = pd.read_csv(processed_dir / "manifest.csv")
    split_df = manifest[manifest["split"] == SPLIT].copy()

    dataset = BrainTumorDataset(
        data_dir=processed_dir,
        manifest_df=split_df,
        transform=get_val_transforms(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("evaluation", {}).get("batch_size", 128),
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"],
        persistent_workers=config["dataloader"]["num_workers"] > 0,
    )

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = create_model(checkpoint.get("config", config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if config["training"].get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    probs, labels = [], []
    for images, lbl in tqdm(dataloader, desc="Inference"):
        images = images.to(device, non_blocking=True)
        if config["training"].get("channels_last", False):
            images = images.to(memory_format=torch.channels_last)
        with autocast(enabled=config["training"]["use_amp"],
                     dtype=torch.float16 if config["training"]["amp_dtype"] == "float16" else torch.bfloat16):
            outputs = model(images)
        probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        labels.extend(lbl.numpy())
    y_proba = np.vstack(probs)
    y_true = np.array(labels)
    y_pred = y_proba.argmax(axis=1)
    conf = y_proba.max(axis=1)

    classes = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))

    # 20-bin confidence histograms
    def hist20(mask):
        bins = np.minimum((conf[mask] * 20).astype(int), 19)
        return np.bincount(bins, minlength=20).tolist()

    correct = y_pred == y_true
    stats: Dict[str, List[int]] = {
        "n_confident_correct": int(correct.sum()),
        "correct_confidence_bins": hist20(correct),
        "n_misclassified": int((~correct).sum()),
        "misclassified_confidence_bins": hist20(~correct),
    }

    # Micro-average ROC (label sanity: classes 0..3, ordered as IDX_TO_CLASS)
    auc_micro = roc_auc_score_all(y_true_bin.ravel(), y_proba.ravel())
    points_micro = downsample(*roc_curve(y_true_bin.ravel(), y_proba.ravel())[:2])
    roc = {"micro_average": {"auc": round(auc_micro, 4), "points": points_micro}}
    for i, name in enumerate(classes):
        auc_i = roc_auc_score_all(y_true_bin[:, i], y_proba[:, i])
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc[name] = {"auc": round(auc_i, 4), "points": downsample(fpr, tpr)}

    # Per-sample CSV for downstream error analysis
    pd.DataFrame({
        "filepath": split_df["filepath"].tolist(),
        "true": [classes[i] for i in y_true],
        "pred": [classes[i] for i in y_pred],
        "correct": correct,
        "confidence": conf,
        **{f"prob_{c}": y_proba[:, j] for j, c in enumerate(classes)},
    }).to_csv(OUT_CSV, index=False)

    summary = {
        "checkpoint": CHECKPOINT,
        "split": SPLIT,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "test_samples_per_class": pd.Series([classes[i] for i in y_true]).value_counts().reindex(classes).tolist(),
        **stats,
        "correct_confidence_bin_sum": int(sum(stats["correct_confidence_bins"])),
        "misclassified_confidence_bin_sum": int(sum(stats["misclassified_confidence_bins"])),
        "roc": roc,
    }
    Path(OUT_JSON).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "roc"}, indent=2))
    print("ROC AUCs:", {k: v["auc"] for k, v in roc.items()})
    print("ROC points sample (micro):", points_micro)


def roc_auc_score_all(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def downsample(fpr: np.ndarray, tpr: np.ndarray, n: int = N_ROC_POINTS):
    """Evenly sample n monotone points from an ROC curve, keep endpoints."""
    idx = [int(round(j * (len(fpr) - 1) / (n - 1))) for j in range(n)]
    pts = [(round(float(fpr[j]), 4), round(float(tpr[j]), 4)) for j in idx]
    pts[0] = [0.0, 0.0]
    return pts


if __name__ == "__main__":
    main()
