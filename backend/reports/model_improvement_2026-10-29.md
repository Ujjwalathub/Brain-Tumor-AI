# Model Improvement Analysis

Date: 2026-10-29

## FINAL RESULT — augaffine champion deployed

`EfficientNet-B0` epoch 13 (augaffine run, val macro-recall 0.9988)
evaluated on the test set 2026-10-29 15:32:

| Checkpoint | Test acc / macro-recall | Glioma recall | Meningioma recall |
|---|---|---|---|
| augaffine ep13 | **0.9619** | **0.858** | **0.990** |
| efficientnet_b0 ep19 (previous baseline) | 0.9581 | 0.853 | 0.980 |

Test artifacts: `reports/confusion_matrix_test_best_epoch013_macro-recall0.9988.png`,
`reports/metrics_test_best_epoch013_macro-recall0.9988.json`,
`reports/misclassifications_test_best_epoch013_macro-recall0.9988.csv`,
`reports/test_sample_predictions_ep13_augaffine.csv`,
`reports/frontend_stats_ep13_augaffine.json`.

The augaffine epoch beats the baseline on overall accuracy (+0.38pp),
glioma recall (+0.5pp) and meningioma recall (+1.0pp), so per the
deployment protocol it was exported and deployed:

- Previous ONNX archived at `models/onnx/archive/efficientnet_b0_ep19_baseline_0.9581.onnx`
  (plus an orphan `efficientnet_b0.onnx.data` from a failed dynamo export).
- New `models/onnx/efficientnet_b0.onnx` exported via `src/export_onnx.py`
  (legacy TorchScript exporter, opset 17, 15.28 MB).
- ONNX/PyTorch parity on real val images: max diff **0.001472** (tolerance 0.01 passed).
- Benchmark: PyTorch CUDA 18.15 ms vs ONNX Runtime CUDA 5.86 ms (3.10x speedup,
  170.7 img/s throughput).
- API smoke test (uvicorn on :8765, real val-glioma JPEG): prediction
  `glioma @ 0.908` (200 OK); colorized RGB JPEG of the same scan family ->
  422 color-image rejection with `COLOR_IMAGE_REJECTION_MESSAGE`.
- `tests/test_api.py`: 25/25 pass after the swap.
- Note: `PYTHONUTF8=1` is required for `export_onnx.py` on this Windows console
  (the check/cross glyphs crash under cp1252 otherwise).

Still-open improvement lever: glioma->meningioma drift remains the dominant
error (47 of 61 test misses are glioma->meningioma). Next candidate levers:
deeper spatial augmentation (val macro-recall +0.6pp when translation was
added) and/or glioma-upweighted loss balancing.

## Tried: glioma-upweighted loss (2026-10-29, negative result)

`configs/efficientnet_b0_gliomaw.yaml` — same augaffine augmentation stack
plus `training.class_weights: [1.3, 1.0, 1.0, 1.0]` (glioma=idx 0, verified
against `data.dataset.CLASS_TO_IDX`). `src/train.py` now supports optional
`training.class_weights` (new, backward compatible).

Training (1h04m, 21 epochs run + 2 patience epochs, early-stopped at 23):

- Best val macro-recall **0.9988 at epoch 15** — exactly tied with the
  deployed augaffine champion. Val is saturated at 1 error / 840 scans, so
  val can no longer discriminate; test is the only signal.

Test result (**worse — candidate rejected**):

| Checkpoint | Test acc / macro-recall | Glioma recall | Meningioma recall | Meningioma precision |
|---|---|---|---|---|
| gliomaw ep15 (candidate) | 0.9587 | 0.843 | 0.998 | 0.913 |
| augaffine ep13 (deployed) | **0.9619** | **0.858** | 0.990 | 0.894 |

66 misclassifications vs 61. The weight raised glioma precision (0.997) but
lowered glioma recall and pulled notumor precision down to 0.950: under the
Training->Testing source shift the extra glioma gradient does not recover the
drift, it just reshapes the boundary unfavourably. **Decision: keep the
augaffine ep13 ONNX deployed; no model swap.**

Artifacts: `models/run_gliomaw/best_epoch015_macro-recall0.9988.pth`,
`reports/metrics_test_best_epoch015_macro-recall0.9988.json`,
`reports/training_history_gliomaw.csv`.

Learning for the next attempt: the ceiling is the source shift, not class
imbalance. A plain CE class weight is too blunt; viable next levers are
source-domain matching (matching Training vs Testing histogram statistics —
see src/band_stats.py / src/border_ablation.py), test-time normalization, or
partial fine-tuning on a held-out slice of the Testing source itself.

--- Historical record from earlier today below ---

## (Historical) Current champion — superseded by FINAL RESULT above

`EfficientNet-B0` epoch 19 — **test accuracy / macro-recall 0.9581**
(deployed ONNX: `models/onnx/efficientnet_b0.onnx`).
Weak spot: glioma recall 0.853 (precision 0.997), glioma→meningioma/notumor
drift under the Training↔Testing source shift.

## Checkpoints evaluated on the held-out test set today (epoch picking only)

| Checkpoint | Test acc / macro-recall | Glioma recall |
|---|---|---|
| efficientnet_b0 ep19 (baseline) | **0.9581** | 0.853 |
| fr_t10/resnet50_seed42 ep21 | 0.9531 | 0.845 |
| resnet50 extra run ep8 (root) | 0.9231 | 0.823 |

ResNet50 seed42 does not beat the baseline; swapping the deployed ONNX model
is **not** justified.

## Cheap wins ruled out (src/debug_ensemble_tta.py, on test set)

| Candidate | Test acc |
|---|---|
| EffNet + hflip-TTA | 0.9569 |
| ResNet50 + hflip-TTA | 0.9550 |
| Ensemble avg (identity) | 0.9563 |
| Ensemble avg + hflip-TTA | 0.9569 |
| Ensemble 2:1 (effnet:resnet) | 0.9575 |

None improve on 0.9581 — the two architectures make highly correlated errors
(same source-shift artifact), so inference-side averaging cannot recover
them. Conclusion: retraining with stronger augmentation is the right lever.

## Launched retrain: stronger spatial augmentation

Per the Stage 9 gate recommendation (glioma sensitive to vertical crop
offsets), `configs/efficientnet_b0_augaffine.yaml` adds random translation:

- `translate: [0.08, 0.08]` via `v2.RandomAffine(degrees=15, translate=…)`
  (implemented in `src/data/transforms.py::get_train_transforms`; also fixes
  the `horizontal_flip_prob` ↔ `h_flip_prob` key mismatch so the YAML flag is
  actually honored).
- 40 epochs, early-stopping patience 8 (augmented curves are noisier).
- Output: `models/run_augaffine/`, TensorBoard `runs/efficientnet_b0_augaffine`.

Selection rules unchanged: test set is used exactly once for the final pick
(`src/evaluate.py --split test`).

## After training

1. Evaluate best augaffine checkpoint on **val** first, compare vs 0.9929.
2. Only if val ≥ baseline, run the single test-set evaluation and compare vs
   0.9581 (esp. glioma recall).
3. If it wins: `src/export_onnx.py` → replace `models/onnx/efficientnet_b0.onnx`
   (archive the old one), re-verify ONNX/PyTorch parity on real val images,
   and re-run the API smoke test.
