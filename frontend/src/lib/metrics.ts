/**
 * Validated clinical & engineering benchmark data for the BrainTumor AI
 * classifier (EfficientNet-B0, augaffine run: best_epoch013_macro-recall0.9988.pth,
 * deployed as models/onnx/efficientnet_b0.onnx).
 *
 * All metrics measured on the held-out test split (1,600 scans,
 * backend/data/processed/manifest.csv split=test) on the RTX 3050 6GB Laptop GPU,
 * regenerated 2026-10-29 via backend/src/generate_frontend_stats.py.
 */

export const CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"] as const;
export type TumorClass = (typeof CLASSES)[number];

export const CLASS_COLORS: Record<string, string> = {
  Glioma: "var(--glioma)",
  Meningioma: "var(--meningioma)",
  "No Tumor": "var(--notumor)",
  Pituitary: "var(--pituitary)",
};

export const BENCHMARK = {
  macroRecall: 0.9619,
  accuracy: 0.9619,
  avgInferenceMs: 5.9,
  vramUsedGb: 0.88,
  vramTotalGb: 6.0,
  params: "~4.0M",
  checkpoint: "best_epoch013_macro-recall0.9988.pth (augaffine run)",
  dataset: "7,200 Kaggle Brain Tumor MRI scans",
} as const;

/** Rows = true class, columns = predicted class (order of CLASSES). Test split, 400 scans per class. */
export const CONFUSION_MATRIX: number[][] = [
  [343, 47, 10, 0],
  [0, 396, 1, 3],
  [0, 0, 400, 0],
  [0, 0, 0, 400],
];

export const TEST_SAMPLES = [400, 400, 400, 400];

export const PER_CLASS = [
  { name: "Glioma", precision: 1.0, recall: 0.8575, f1: 0.9233 },
  { name: "Meningioma", precision: 0.8939, recall: 0.99, f1: 0.9395 },
  { name: "No Tumor", precision: 0.9732, recall: 1.0, f1: 0.9864 },
  { name: "Pituitary", precision: 0.9926, recall: 1.0, f1: 0.9963 },
] as const;

export const MACRO = { precision: 0.9599, recall: 0.9619, f1: 0.9614 } as const;

/** Epochs 1-21. Stage 1 (1-5) head-only; Stage 2 (6-21) full fine-tune. */
export const TRAINING_CURVES = {
  trainLoss: [1.319, 1.094, 0.909, 0.854, 0.829, 0.584, 0.474, 0.423, 0.403, 0.386, 0.372, 0.369, 0.363, 0.363, 0.361, 0.361, 0.359, 0.357, 0.358, 0.356, 0.357],
  valLoss: [1.25, 1.051, 0.934, 0.925, 0.877, 0.481, 0.424, 0.405, 0.387, 0.377, 0.379, 0.364, 0.363, 0.361, 0.368, 0.36, 0.364, 0.36, 0.36, 0.358, 0.361],
  valRecall: [0.682, 0.69, 0.749, 0.714, 0.782, 0.946, 0.971, 0.981, 0.985, 0.988, 0.985, 0.996, 0.999, 0.999, 0.99, 0.995, 0.995, 0.995, 0.995, 0.996, 0.994],
  stage1End: 5,
  learningRates: ["1e-3", "1e-3", "1e-3", "1e-3", "1e-3", "3e-4", "3e-4", "2.9e-4", "2.9e-4", "2.8e-4", "2.8e-4", "2.7e-4", "2.6e-4", "2.5e-4", "2.4e-4", "2.3e-4", "2.2e-4", "2e-4", "1.9e-4", "1.8e-4", "1.6e-4"],
} as const;

/** 20 bins over 0.0-1.0 confidence. Correct predictions (1,539 of 1,600 test scans). */
export const CORRECT_CONFIDENCE_BINS = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 3, 3, 4, 3, 4, 7, 11, 189, 1190, 123];

/** Misclassified scans — the model's honest error set (61 of 1,600 test scans), by predicted confidence. */
export const AMBIGUOUS_CONFIDENCE_BINS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 3, 1, 2, 4, 11, 3, 4, 14, 13, 2];

/** One-vs-Rest ROC sample points (FPR, TPR) per class, measured on the test split. */
export const ROC_CURVES: { name: string; auc: number; points: [number, number][] }[] = [
  { name: "Glioma", auc: 0.9586, points: [[0, 0], [0, 0.4075], [0, 0.8], [0.1, 0.935], [0.2075, 0.94], [0.3358, 0.945], [0.4425, 0.96], [0.5667, 0.9625], [0.6583, 0.9725], [0.7617, 0.9725], [0.8692, 0.9725], [1, 1]] },
  { name: "Meningioma", auc: 0.9909, points: [[0, 0], [0.0075, 0.27], [0.01, 0.82], [0.035, 0.985], [0.1733, 1], [0.2942, 1], [0.4233, 1], [0.5325, 1], [0.6267, 1], [0.7117, 1], [0.8092, 1], [1, 1]] },
  { name: "No Tumor", auc: 0.9992, points: [[0, 0], [0.0008, 0.5775], [0.005, 1], [0.16, 1], [0.26, 1], [0.3425, 1], [0.4308, 1], [0.54, 1], [0.6433, 1], [0.7292, 1], [0.8383, 1], [1, 1]] },
  { name: "Pituitary", auc: 0.9999, points: [[0, 0], [0, 0.3975], [0, 0.8475], [0.1308, 1], [0.23, 1], [0.3525, 1], [0.4367, 1], [0.535, 1], [0.6433, 1], [0.7483, 1], [0.8458, 1], [1, 1]] },
  { name: "Micro-Average", auc: 0.9869, points: [[0, 0], [0.0031, 0.8369], [0.034, 0.9788], [0.1194, 0.9844], [0.2304, 0.985], [0.3321, 0.9862], [0.4485, 0.99], [0.58, 0.9912], [0.6979, 0.9931], [0.8131, 0.9931], [0.9085, 0.9962], [1, 1]] },
];

/**
 * Multi-model comparative evaluation on the RTX 3050 6GB.
 * EfficientNet-B0 and ResNet50 rows are measured on the test split (2026-10-29);
 * the ConvNeXt-Tiny row carries over the earlier stage-spec evaluation protocol and
 * has not been re-run on the current test split.
 */
export const MODEL_COMPARISON = [
  { name: "EfficientNet-B0 (Deployed)", params: "4.0M", batch: 32, gradAccum: 2, macroRecall: 0.9619, accuracy: 0.9619, latencyMs: 5.9, vramGb: 0.9, deployed: true },
  { name: "ResNet50", params: "25.6M", batch: 32, macroRecall: 0.9531, accuracy: 0.9531, latencyMs: 23.8, vramGb: 5.4, deployed: false },
  { name: "ConvNeXt-Tiny", params: "28.6M", batch: 24, gradAccum: 2, macroRecall: 0.964, accuracy: 0.9675, latencyMs: 27.2, vramGb: 5.6, deployed: false },
] as const;

export const formatPct = (v: number, digits = 2) => `${(v * 100).toFixed(digits)}%`;
