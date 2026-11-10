"""
FR-T10 comparison driver: ResNet50 + ConvNeXt-Tiny, >=3 seeds each.

Runs six sequential trainings (2 models x 3 seeds) and keeps artifacts
separable:
  - generates per-seed configs under configs/fr_t10/ (inherit chain)
  - archives flat checkpoints into models/archive_efficientnet_b0/ before
    starting (the baseline checkpoint is fully consumed: Stage 10 done,
    ONNX exported)
  - after each run copies reports/training_history.csv to
    reports/fr_t10/{model}_seed{seed}_history.csv and moves the run's
    checkpoint files into models/fr_t10/{model}_seed{seed}/
  - prints a final summary table of best val macro-recall per run

Usage (run from backend/):
    ..\\.venv\\Scripts\\python.exe src\\run_fr_t10.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

SEEDS = [42, 43, 44]
MODELS = ["resnet50", "convnext_tiny"]


def write_seed_config(model: str, seed: int, configs_dir: Path) -> Path:
    """Create a per-seed config that inherits the model config."""
    out_path = configs_dir / "fr_t10" / f"{model}_seed{seed}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"# Auto-generated for FR-T10 comparison\n"
        f"# NOTE: _base_ paths resolve relative to THIS file's directory\n"
        f"_base_: ../{model}.yaml\n\n"
        f"seed: {seed}\n"
        f"experiment_name: \"{model}_seed{seed}\"\n",
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    backend = Path.cwd()
    configs_dir = backend / "configs"
    models_dir = backend / "models"
    reports_dir = backend / "reports"
    history_csv = reports_dir / "training_history.csv"

    venv_python = (backend.parent / ".venv" / "Scripts" / "python.exe").resolve()
    if not venv_python.exists():
        raise SystemExit(f"venv python not found: {venv_python}")

    # ------------------------------------------------------------------
    # Archive flat baseline checkpoints so run outputs stay separable
    # ------------------------------------------------------------------
    archive = models_dir / "archive_efficientnet_b0"
    existing_ckpts = sorted(models_dir.glob("*.pth"))
    if existing_ckpts and not archive.exists():
        archive.mkdir(parents=True, exist_ok=True)
        for ckpt in existing_ckpts:
            shutil.move(str(ckpt), str(archive / ckpt.name))
        print(f"Archived {len(existing_ckpts)} baseline checkpoints -> {archive}")
    else:
        print("No flat checkpoints to archive (or already archived)")

    results = []
    for model in MODELS:
        for seed in SEEDS:
            tag = f"{model}_seed{seed}"
            config_path = write_seed_config(model, seed, configs_dir)
            print("=" * 70, flush=True)
            print(f"TRAINING: {tag} (config: {config_path.name})", flush=True)
            print("=" * 70, flush=True)

            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            proc = subprocess.run(
                [str(venv_python), "src/train.py", "--config", str(config_path)],
                cwd=str(backend),
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                print(f"!! {tag} training FAILED (rc={proc.returncode})", flush=True)
                results.append((tag, "FAILED", None))
                continue

            # Preserve training history before the next run overwrites it
            fr_t10_reports = reports_dir / "fr_t10"
            fr_t10_reports.mkdir(parents=True, exist_ok=True)
            if history_csv.exists():
                shutil.copy2(history_csv, fr_t10_reports / f"{tag}_history.csv")

            # Move this run's checkpoint files into a per-run folder
            run_dir = models_dir / "fr_t10" / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for ckpt in sorted(models_dir.glob("*.pth")):
                dest = run_dir / ckpt.name
                if dest.exists():
                    continue  # never overwrite across seeds
                shutil.move(str(ckpt), str(dest))
                moved += 1
            print(f"{tag}: moved {moved} checkpoints -> {run_dir}", flush=True)

            # Best val metric from the checkpoint filenames
            best_metric = None
            for ckpt in sorted(run_dir.glob("best_epoch*.pth")):
                try:
                    metric = float(ckpt.stem.split("macro-recall")[-1])
                    best_metric = max(best_metric or 0.0, metric)
                except ValueError:
                    pass
            results.append((tag, "OK", best_metric))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("FR-T10 comparison — best VALIDATION macro-recall per run")
    print("=" * 70)
    for tag, status, metric in results:
        metric_str = f"{metric:.4f}" if metric is not None else "-"
        print(f"  {tag:<26} {status:<6} {metric_str}")
    print()


if __name__ == "__main__":
    import os

    main()
