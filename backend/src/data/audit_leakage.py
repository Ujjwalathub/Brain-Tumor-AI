#!/usr/bin/env python3
"""
Dataset leakage audit.

The Kaggle Brain Tumor MRI dataset contains byte-identical images that appear
in more than one official split. When a manifest puts any of those scans in
both train and test, the reported test metrics are optimistic.

This script hashes every cached image referenced by ``manifest.csv`` and
reports:

* duplicate groups (same bytes, different filepath),
* groups spanning train/val/test,
* the exact number of test scans whose content appears in train or val,
* the accuracy impact estimate (optional, requires predictions).

Usage:
    python src/data/audit_leakage.py
    python src/data/audit_leakage.py --check          # exit 1 on leakage
    python src/data/audit_leakage.py --write-clean    # also write manifest_clean.csv
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import configure_utf8_console, load_config  # noqa: E402
from src.data.split import content_hash  # noqa: E402

configure_utf8_console()

SPLIT_PAIRS = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]


def audit(processed_dir: Path) -> dict:
    manifest_path = processed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    hashes = [content_hash(processed_dir / p) for p in manifest["filepath"]]
    manifest = manifest.assign(content_hash=hashes)

    groups = defaultdict(list)
    for idx, row in manifest.iterrows():
        groups[row["content_hash"]].append(idx)

    duplicate_groups = {h: idxs for h, idxs in groups.items() if len(idxs) > 1}
    cross_split = {
        h: idxs
        for h, idxs in duplicate_groups.items()
        if manifest.loc[idxs, "split"].nunique() > 1
    }

    pair_counts = {}
    for a, b in SPLIT_PAIRS:
        pair_counts[f"{a}-{b}"] = sum(
            1
            for idxs in duplicate_groups.values()
            if {a, b}.issubset(set(manifest.loc[idxs, "split"]))
        )

    test_hashes_in_train = set(
        manifest.loc[manifest["split"] == "test", "content_hash"]
    ) & set(manifest.loc[manifest["split"] == "train", "content_hash"])
    test_hashes_in_val = set(
        manifest.loc[manifest["split"] == "test", "content_hash"]
    ) & set(manifest.loc[manifest["split"] == "val", "content_hash"])

    test_row = manifest["split"] == "test"
    leaked_test_rows = manifest.loc[
        test_row & manifest["content_hash"].isin(test_hashes_in_train | test_hashes_in_val)
    ]

    return {
        "manifest": str(manifest_path),
        "total_rows": int(len(manifest)),
        "unique_content": int(manifest["content_hash"].nunique()),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_rows": int(len(manifest) - manifest["content_hash"].nunique()),
        "cross_split_groups": len(cross_split),
        "cross_split_rows": int(
            sum(len(idxs) for idxs in cross_split.values())
        ),
        "group_counts_by_pair": pair_counts,
        "test_rows": int(test_row.sum()),
        "leaked_test_rows": int(len(leaked_test_rows)),
        "clean_test_rows": int(test_row.sum() - len(leaked_test_rows)),
        "leaked_test_by_class": (
            leaked_test_rows["class"].value_counts().to_dict()
            if len(leaked_test_rows)
            else {}
        ),
        "_manifest": manifest,
        "_leaked_index": leaked_test_rows.index,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit manifest for content leakage")
    parser.add_argument(
        "--config",
        default=str(project_root / "configs" / "efficientnet_b0_augaffine.yaml"),
        help="Config used to locate data.processed_dir",
    )
    parser.add_argument("--check", action="store_true", help="Exit 1 if leakage is found")
    parser.add_argument(
        "--write-clean",
        action="store_true",
        help="Write manifest_clean.csv without leaked test rows",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config["data"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir

    report = audit(processed_dir)
    manifest = report.pop("_manifest")
    leaked_index = report.pop("_leaked_index")

    print("=" * 70)
    print("Dataset Leakage Audit")
    print("=" * 70)
    print(f"Manifest:            {report['manifest']}")
    print(f"Rows:                {report['total_rows']:,}")
    print(f"Unique content:      {report['unique_content']:,}")
    print(f"Duplicate groups:    {report['duplicate_groups']:,}")
    print(f"Cross-split groups:  {report['cross_split_groups']:,}")
    for pair, count in report["group_counts_by_pair"].items():
        print(f"  {pair:12s}: {count:,}")
    print(
        f"Test scans leaked:   {report['leaked_test_rows']:,} / "
        f"{report['test_rows']:,} ({report['leaked_test_rows'] / max(report['test_rows'], 1):.1%})"
    )
    if report["leaked_test_by_class"]:
        print(f"  by class:          {report['leaked_test_by_class']}")
    print(f"Leak-free test rows: {report['clean_test_rows']:,}")

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "leakage_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to:   {out_path}")

    if args.write_clean:
        clean_path = processed_dir / "manifest_clean.csv"
        manifest.drop(index=leaked_index).drop(columns=["content_hash"]).to_csv(
            clean_path, index=False
        )
        print(f"Clean manifest:      {clean_path}")

    if report["leaked_test_rows"] > 0:
        print(
            "\n[WARN] Test metrics computed with this manifest are inflated. "
            "Re-run split.py (content-aware) or evaluate with the clean manifest."
        )
        return 1 if args.check else 0

    print("\n[OK] No test leakage detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
