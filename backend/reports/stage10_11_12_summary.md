# Stage 10–12 Summary — EfficientNet-B0 Baseline

Date: 2026-10-26

## Stage 10: Test evaluation (test set used exactly once)
Checkpoint: `models/best_epoch019_macro-recall0.9929.pth` (epoch 19, val macro-recall 0.9929)

| Class | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| glioma | 0.997 | 0.853 | 0.919 | 0.957 |
| meningioma | 0.920 | 0.980 | 0.949 | 0.993 |
| notumor | 0.935 | 1.000 | 0.966 | 0.998 |
| pituitary | 0.990 | 1.000 | 0.995 | 1.000 |

Overall accuracy / macro-recall: **0.9581** on 1,600 held-out test images.
- Gap vs validation (0.9929) is consistent with the known source-shift between
  this benchmark's Training/ and Testing/ folders; glioma recall (0.853) with
  0.997 precision matches published baselines and shows as glioma→meningioma
  confusion at high confidence.
- Artifacts: `reports/metrics_test_*.json`, `reports/confusion_matrix_test_*.png`,
  `reports/roc_curves_test_*.png`, `reports/misclassifications_test_*.csv`.

## Stage 11: ONNX export & parity
- Artifact: `models/efficientnet_b0.onnx` (15.28 MB, opset 17, legacy TorchScript
  exporter — the torch≥2.14 dynamo default dropped weights, caveat noted in code).
- Parity: max diff 0.001 / mean 0.0004 vs PyTorch on 4 real val images.
- **Parity-check lesson:** synthetic inputs (gaussian/uniform) are invalid
  parity probes for this model — trained BN stats amplify OOD inputs so much
  that CPU/GPU conv differences flip degenerate logits (~±3000). Real inputs
  agree to <1e-3 across CPU fp32 / GPU fp32 / GPU fp16 / ORT.
- Performance: ORT-CUDA 7.9 ms/img vs PyTorch-CUDA 23.6 ms/img (≈3×).

## Stage 12: API smoke test
- Server: `uvicorn api.main:app` (127.0.0.1:8000), model auto-discovered in
  `models/onnx/`.
- `/health` → healthy, active provider `CUDAExecutionProvider`.
- `/predict` on val image Tr-gl_10 → glioma @ 94.5%, ~26 ms end-to-end.
- Deployment gotcha fixed in `api/inference.py`: ORT's CUDA EP cannot find
  cuDNN on Windows unless `torch/lib` + `onnxruntime/capi` are registered via
  `os.add_dll_directory()` **and** `PATH` (the cuDNN 9 facade resolves its
  sub-DLLs by name through the full search chain). With only
  `add_dll_directory`, startup fails with "Invalid handle. Cannot load symbol
  cudnnCreate".

## Pipeline status
Stages 1–12 complete for the EfficientNet-B0 baseline.
Open items: FR-T10 comparison runs (ResNet50 + ConvNeXt-Tiny, ≥3 seeds) and
Stage 13 final documentation.
