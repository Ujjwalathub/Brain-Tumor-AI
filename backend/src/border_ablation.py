"""
Border Ablation Test - Stage 9 diagnostic
==========================================

QUESTION: Does the model's decision depend on the image borders / facial
region, or only on the central brain content?

Grad-CAM (correlational) showed margin-dominant activation for pituitary
in particular. This script provides the CAUSAL test: evaluate the model on
the full validation split with border bands of the image masked out (filled
with black, i.e. as if that band were cropped away) and compare per-class
accuracy against the unmasked baseline.

If border masking collapses pituitary accuracy -> the model really uses
border information (shortcut). If accuracy holds -> the Grad-CAM margin
heat was incidental, not causal.

Usage (run from backend/):
    ..\\.venv\\Scripts\\python.exe src\\border_ablation.py \
        --checkpoint models\\best_epoch019_macro-recall0.9929.pth \
        --config configs\\efficientnet_b0.yaml --split val
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from models import create_model
from utils import load_config, set_seed
from data.dataset import CLASS_TO_IDX
from data.transforms import get_val_transforms


# ---------------------------------------------------------------------------
# Mask variants: (name, description, band thickness as fraction of side)
# The bottom band is where the facial / peri-orbital tissue lives in coronal
# scout MRIs (the pituitary class presumed shortcut); top/left/right bands
# complete the check.
# ---------------------------------------------------------------------------
MASK_VARIANTS: List[Tuple[str, str]] = [
    ("none", "Baseline, no masking"),
    ("bottom", "Bottom 20% rows zeroed (facial band)"),
    ("top", "Top 20% rows zeroed"),
    ("top_bottom", "Top and bottom 20% zeroed"),
    ("all_borders", "All four borders, 10% each, zeroed"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Border-ablation shortcut test for the MRI classifier"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Split to evaluate (default: val — test stays locked)")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def apply_mask(image: np.ndarray, variant: str) -> np.ndarray:
    """Zero the specified border bands of a uint8 (H, W, C) image."""
    h, w = image.shape[:2]
    masked = image.copy()

    if variant == "none":
        return masked
    elif variant == "bottom":
        masked[int(h * 0.8):, :, :] = 0
        return masked
    elif variant == "top":
        masked[: int(h * 0.2)] = 0
        return masked
    elif variant == "top_bottom":
        masked[: int(h * 0.2)] = 0
        masked[int(h * 0.8):, :, :] = 0
        return masked
    elif variant == "all_borders":
        masked[: int(h * 0.1)] = 0
        masked[int(h * 0.9):] = 0
        masked[:, : int(w * 0.1)] = 0
        masked[:, int(w * 0.9):] = 0
        return masked
    raise ValueError(f"Unknown mask variant: {variant}")


def evaluate_variant(
    variant: str,
    model: torch.nn.Module,
    df: pd.DataFrame,
    processed_dir: Path,
    transform,
    device: str,
    batch_size: int,
    channels_last: bool,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate masked accuracy per class for one mask variant."""
    n_classes = max(CLASS_TO_IDX.values()) + 1
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    model.eval()
    batch_images: List[torch.Tensor] = []
    batch_labels: List[int] = []

    filepaths = df["filepath"].tolist()
    labels = df["class"].tolist()

    with torch.no_grad():
        def flush():
            if not batch_images:
                return
            batch = torch.stack(batch_images).to(device)
            if channels_last:
                batch = batch.to(memory_format=torch.channels_last)
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            for label, pred in zip(batch_labels, preds):
                confusion[label, pred] += 1
            batch_images.clear()
            batch_labels.clear()

        for filepath, class_name in zip(filepaths, labels):
            image = np.load(processed_dir / filepath)
            image = apply_mask(image, variant)
            tensor = transform(image)
            batch_images.append(tensor)
            batch_labels.append(CLASS_TO_IDX[class_name])
            if len(batch_images) >= batch_size:
                flush()
        flush()

    per_class_acc = {}
    for idx, class_name in enumerate(
        [k for k, v in CLASS_TO_IDX.items() if v < n_classes]
    ):
        total = confusion[idx].sum()
        correct = confusion[idx, idx]
        per_class_acc[class_name] = correct / total if total else float("nan")
    return per_class_acc, confusion


def main() -> None:
    print("=" * 70)
    print("Border Ablation Test (Stage 9 shortcut diagnostic)")
    print("=" * 70)
    assert torch.cuda.is_available(), "CUDA GPU required. Aborting."
    device = "cuda"

    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    processed_dir = Path.cwd() / config["data"]["processed_dir"]
    manifest = pd.read_csv(processed_dir / "manifest.csv")
    df = manifest[manifest["split"] == args.split].reset_index(drop=True)
    print(f"Split: {args.split} | Images: {len(df)}")
    print(df["class"].value_counts().to_string())

    # Load model from checkpoint (weights_only=False: local trusted file
    # containing numpy metrics).
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model = create_model(checkpoint.get("config", config))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if config["training"].get("channels_last", False):
        model = model.to(memory_format=torch.channels_last)

    transform = get_val_transforms(config)
    use_amp = config["training"]["use_amp"]
    amp_dtype = (
        torch.float16
        if config["training"]["amp_dtype"] == "float16"
        else torch.bfloat16
    )
    autocast = torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)

    results: Dict[str, Dict[str, float]] = {}
    confusions: Dict[str, np.ndarray] = {}
    for name, description in MASK_VARIANTS:
        print(f"Variant: {name} — {description}")
        with autocast:
            accs, conf = evaluate_variant(
                name, model, df, processed_dir, transform,
                device, args.batch_size,
                config["training"].get("channels_last", False),
            )
        confusions[name] = conf
        results[name] = accs
        for cls, acc in accs.items():
            print(f"  {cls:<12}: {acc * 100:6.2f}%")

    # ------------------------------------------------------------------
    # Summary table with deltas vs. baseline
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY (accuracy %, delta vs baseline)")
    print("=" * 70)
    classes = config["data"]["classes"]
    header = "  {:<12}".format("variant") + "".join(
        f"{c:>14}" for c in classes
    )
    print(header)
    baseline = results["none"]
    for name, description in MASK_VARIANTS:
        cells = []
        for c in classes:
            acc = results[name][c] * 100
            delta = results[name][c] - baseline[c]
            cells.append(f"{acc:6.2f} ({delta:+.2%})")
        print("  {:<12}".format(name) + "".join(f"{c:>14}" for c in cells))

    # Confusion matrices show WHERE masked predictions migrate to
    print()
    print("=" * 70)
    print("CONFUSION MATRICES (rows = true, cols = predicted)")
    print("=" * 70)
    for name, _ in MASK_VARIANTS:
        print(f"\n[{name}]")
        print("  " + "".join(f"{c[:8]:>10}" for c in classes))
        for idx, c in enumerate(classes):
            row = confusions[name][idx]
            print("  {:<9}".format(c[:9]) + "".join(f"{v:>10}" for v in row))

    # Save report
    reports_dir = Path.cwd() / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "border_ablation.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Border Ablation Test Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split} | Images: {len(df)}\n\n")
        for name, description in MASK_VARIANTS:
            f.write(f"[{name}] {description}\n")
            for c in classes:
                f.write(f"  {c:<12}: {results[name][c] * 100:6.2f}%\n")
            f.write("\n")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
