# Stage 9 Gate Decision — Explainability / Shortcut Audit

Date: 2026-10-26
Checkpoint: `models/best_epoch019_macro-recall0.9929.pth` (EfficientNet-B0, epoch 19)
Data used: **validation split only** (840 images, 210/class) — test set untouched.

## Methods used
1. Grad-CAM (20 val images, 5/class) and Grad-CAM++ (40 images, 10/class).
2. Causal border-ablation test (`src/border_ablation.py`): full val split with
   border bands zeroed — [top 20%] / [bottom 20% FACE band] / [top+bottom]
   / [all borders 10%].
3. Band-content statistics (`src/band_stats.py`): mean brightness/tissue
   fraction per band per class (occlusion-vs-shortcut discriminator).

## Findings
- Grad-CAM/Grad-CAM++ flagged 12/20 and 36/40 "suspicious" cases via a
  center-vs-margin heuristic. Panel-by-panel review: meningioma heat lands on
  the lesion (5/5 + 10/10 correct); notumor heat is diffuse over brain tissue
  and *never* fixates the bright skull ring (the known Br35H single-source
  risk); pituitary shows repeated margin/facial-band heat (9/10 flagged).
- Causal ablation **refuted** the pituitary border concern:
  | variant        | glioma | meningioma | notumor | pituitary |
  |----------------|--------|-----------|---------|-----------|
  | none           | 99.05% | 98.10%    | 100.00% | 100.00%   |
  | bottom (face)  | 98.10% | 98.10%    | 100.00% | 100.00%   |
  | top            | 91.43% | 96.67%    | 100.00% | 100.00%   |
  | top + bottom   | 69.05% | 96.19%    | 100.00% |  99.52%   |
  | all borders    | 76.67% | 98.10%    | 100.00% |  98.10%   |
- Pituitary retains ~100% accuracy with the entire facial band erased → the
  margin Grad-CAM heat is incidental attention, not decision-relevant.
- Glioma *is* sensitive to band masking (−30pp). Confusion migration spreads
  across meningioma (22) and notumor (41) with no fixed artifact target, and
  band brightness is statistically identical across all four classes
  (band_stats.csv) → best explanation: masking occludes lesion evidence of
  large infiltrative tumors, not a border-artifact shortcut.

## Decision: GATE PASSED
No systematic border/background artifact reliance drives class decisions.
Both historic concerns (pituitary facial-band heat; notumor single-source
Br35H shortcut) are causally discharged.

## Documented caveat (does not gate Stage 10)
- Glioma decisions degrade under aggressive vertical cropping (69% with 40%
  of vertical content removed). This is an input-cropping robustness risk,
  not leakage. Mitigation for any future retrain: stronger translation /
  random-crop augmentation.
- Confidence calibration note: masking changes input statistics (black
  bands are OOD); deltas were judged by *misclassification direction*, not
  absolute magnitude.
