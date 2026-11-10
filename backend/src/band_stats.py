"""
Band Content Statistics - companion to border_ablation.py
==========================================================

border_ablation.py found that masking the top/bottom 20% bands drops
glioma accuracy by ~30pp while pituitary/notumor are unaffected. Two
hypotheses:

  A) Shortcut: glioma decisions use border-specific artifacts.
  B) Occlusion: glioma images have bright tissue (lesion/brain) extending
     into those bands, so masking removes real tumor evidence.

This script measures image intensity in the top and bottom 20% bands per
class on a split. If glioma bands hold much more bright tissue than the
other classes', hypothesis B is favored: masking cut through anatomy, not
artifacts.

Usage (run from backend/):
    ..\\.venv\\Scripts\\python.exe src\\band_stats.py \
        --config configs\\efficientnet_b0.yaml --split val
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from utils import load_config
from data.dataset import CLASS_TO_IDX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brightness statistics of border bands per class"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--band-frac", type=float, default=0.2,
                        help="Band thickness as fraction of image height")
    return parser.parse_args()


def band_stats(image: np.ndarray, band_frac: float) -> Dict[str, float]:
    """Fraction of 'tissue' (>20 intensity) pixels in top and bottom bands."""
    h = image.shape[0]
    gray = image.mean(axis=2) if image.ndim == 3 else image
    top_band = gray[: int(h * band_frac)]
    bottom_band = gray[int(h * (1.0 - band_frac)):]

    def tissue(band: np.ndarray) -> float:
        return float(band.mean() / 255.0)

    return {
        "tissue_frac_top": float((top_band > 20).mean()),
        "tissue_frac_bottom": float((bottom_band > 20).mean()),
        "mean_top": tissue(top_band),
        "mean_bottom": tissue(bottom_band),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    processed_dir = Path.cwd() / config["data"]["processed_dir"]

    manifest = pd.read_csv(processed_dir / "manifest.csv")
    df = manifest[manifest["split"] == args.split].reset_index(drop=True)
    print(f"Split: {args.split} | Images: {len(df)} | Band frac: {args.band_frac}")

    rows = []
    for filepath, class_name in zip(df["filepath"], df["class"]):
        image = np.load(processed_dir / filepath)
        stats = band_stats(image, args.band_frac)
        rows.append({"class": class_name, **stats})
    stats_df = pd.DataFrame(rows)

    agg = stats_df.groupby("class").mean().reindex(
        [k for k, _ in sorted(CLASS_TO_IDX.items(), key=lambda kv: kv[1])]
    )
    print()
    print("Mean fraction of image brightness per band (higher = more tissue)")
    print(agg.to_string(float_format=lambda v: f"{v:6.3f}"))

    out_path = Path.cwd() / "reports" / "band_stats.csv"
    agg.to_csv(out_path, float_format="%.5f")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
