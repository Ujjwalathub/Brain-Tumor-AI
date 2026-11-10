import { createFileRoute } from "@tanstack/react-router";
import {
  Activity,
  Award,
  Zap,
  HardDrive,
  BarChart3,
  TrendingUp,
  LineChart,
  ShieldAlert,
  Server,
  Layers,
} from "lucide-react";
import {
  ConfusionMatrixChart,
  MetricsBars,
  MetricsRadar,
  RadarLegend,
  TrainingCurves,
  ConfidenceHistogram,
  RocCurves,
  RocLegend,
  ModelComparisonTable,
  GpuMonitor,
} from "@/components/charts";
import { BENCHMARK } from "@/lib/metrics";

export const Route = createFileRoute("/performance")({
  component: PerformancePage,
});

export function PerformancePage() {
  return (
    <div className="space-y-16 py-6 md:py-10">
      {/* ================= PAGE HEADER ================= */}
      <section className="container mx-auto px-4 text-center md:px-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/60 px-3.5 py-1.5 shadow-sm backdrop-blur-md">
          <span className="h-2 w-2 rounded-full bg-[var(--teal-soft)] animate-pulse" />
          <span className="font-mono text-xs font-semibold text-ink-deep">
            Empirical Validation · 7,200 Kaggle Brain Tumor MRI Dataset
          </span>
        </div>
        <h1 className="mt-4 font-display text-3xl font-extrabold tracking-tight text-ink-deep sm:text-4xl lg:text-5xl">
          Model Performance & Clinical Analytics
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-base text-ink-body md:text-lg">
          Complete scientific and engineering transparency. Empirical evaluation of the production
          EfficientNet-B0 architecture trained with PyTorch 2.x and ONNX Runtime GPU acceleration.
        </p>
      </section>

      {/* ================= 4 BENCHMARK BASELINE METRIC CARDS ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="liquid-card radius-card-md p-6">
            <div className="flex items-center justify-between text-[var(--teal-soft)]">
              <Award size={24} />
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-ink-muted">
                Test Set
              </span>
            </div>
            <div className="mt-4">
              <span className="font-mono text-3xl font-extrabold text-ink-deep">
                {(BENCHMARK.macroRecall * 100).toFixed(2)}%
              </span>
              <h3 className="mt-1 font-display text-sm font-bold text-ink-deep">
                Macro-Averaged Recall
              </h3>
              <p className="mt-1 text-xs text-ink-muted">
                Near-zero false negatives. Critical for early patient safety.
              </p>
            </div>
          </div>

          <div className="liquid-card radius-card-md p-6">
            <div className="flex items-center justify-between text-[var(--cyan-liquid)]">
              <Activity size={24} />
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-ink-muted">
                Overall
              </span>
            </div>
            <div className="mt-4">
              <span className="font-mono text-3xl font-extrabold text-ink-deep">
                {(BENCHMARK.accuracy * 100).toFixed(2)}%
              </span>
              <h3 className="mt-1 font-display text-sm font-bold text-ink-deep">
                4-Class Accuracy
              </h3>
              <p className="mt-1 text-xs text-ink-muted">
                Balanced precision across 1,600 unseen clinical test scans.
              </p>
            </div>
          </div>

          <div className="liquid-card radius-card-md p-6">
            <div className="flex items-center justify-between text-[var(--pituitary)]">
              <Zap size={24} />
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-ink-muted">
                ONNX GPU
              </span>
            </div>
            <div className="mt-4">
              <span className="font-mono text-3xl font-extrabold text-ink-deep">
                {BENCHMARK.avgInferenceMs} ms
              </span>
              <h3 className="mt-1 font-display text-sm font-bold text-ink-deep">
                Average Latency
              </h3>
              <p className="mt-1 text-xs text-ink-muted">
                High-throughput screening on NVIDIA RTX 3050 Laptop GPU.
              </p>
            </div>
          </div>

          <div className="liquid-card radius-card-md p-6">
            <div className="flex items-center justify-between text-[var(--meningioma)]">
              <HardDrive size={24} />
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-ink-muted">
                Efficiency
              </span>
            </div>
            <div className="mt-4">
              <span className="font-mono text-3xl font-extrabold text-ink-deep">
                {BENCHMARK.vramUsedGb} GB
              </span>
              <h3 className="mt-1 font-display text-sm font-bold text-ink-deep">
                Peak VRAM Usage
              </h3>
              <p className="mt-1 text-xs text-ink-muted">
                Measured peak on the 6GB RTX 3050 — ample headroom for concurrent workloads.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ================= CHARTS GRID 1: CONFUSION MATRIX & PER-CLASS BARS ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="grid gap-8 lg:grid-cols-12">
          {/* Confusion Matrix (6 cols) */}
          <div className="liquid-card radius-card-lg p-6 lg:col-span-6">
            <div className="flex items-center justify-between border-b border-white/60 pb-3">
              <div className="flex items-center gap-2">
                <BarChart3 size={19} className="text-[var(--cyan-liquid)]" />
                <h2 className="font-display text-base font-bold text-ink-deep">
                  4×4 Liquid Confusion Matrix Heatmap
                </h2>
              </div>
              <span className="font-mono text-[0.68rem] text-ink-muted">1,600 Test Scans</span>
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              Y-axis represents true clinical diagnosis; X-axis represents model prediction.
              Hover any cell to inspect exact sample distributions.
            </p>
            <div className="mt-6 flex justify-center">
              <ConfusionMatrixChart />
            </div>
          </div>

          {/* Per-Class Metrics Bar & Radar (6 cols) */}
          <div className="liquid-card radius-card-lg p-6 lg:col-span-6">
            <div className="flex items-center justify-between border-b border-white/60 pb-3">
              <div className="flex items-center gap-2">
                <TrendingUp size={19} className="text-[var(--teal-soft)]" />
                <h2 className="font-display text-base font-bold text-ink-deep">
                  Per-Class Precision, Recall & F1-Score
                </h2>
              </div>
              <span className="font-mono text-[0.68rem] text-ink-muted">Macro: 96.19%</span>
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              Evaluated with balanced weighting across all four cranial tissue states.
            </p>
            <div className="mt-6">
              <MetricsBars />
            </div>
            <div className="mt-8 border-t border-white/60 pt-4">
              <h3 className="text-center font-display text-xs font-bold text-ink-deep">
                Multi-Axis Spider Radar Distribution
              </h3>
              <div className="mt-2 flex justify-center">
                <MetricsRadar />
              </div>
              <div className="mt-3">
                <RadarLegend />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================= CHARTS GRID 2: TRAINING CURVES & CONFIDENCE HISTOGRAM ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="grid gap-8 lg:grid-cols-12">
          {/* Training Curves (7 cols) */}
          <div className="liquid-card radius-card-lg p-6 lg:col-span-7">
            <div className="flex items-center justify-between border-b border-white/60 pb-3">
              <div className="flex items-center gap-2">
                <LineChart size={19} className="text-[var(--pituitary)]" />
                <h2 className="font-display text-base font-bold text-ink-deep">
                  Two-Stage Training & Loss Dynamics
                </h2>
              </div>
              <span className="font-mono text-[0.68rem] text-ink-muted">21 Epochs</span>
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              Stage 1 (Epochs 1–5): Frozen backbone head training. Stage 2 (Epochs 6–21): Full
              fine-tune with Cosine Annealing LR scheduling.
            </p>
            <div className="mt-6">
              <TrainingCurves />
            </div>
          </div>

          {/* Confidence Histogram (5 cols) */}
          <div className="liquid-card radius-card-lg p-6 lg:col-span-5">
            <div className="flex items-center justify-between border-b border-white/60 pb-3">
              <div className="flex items-center gap-2">
                <ShieldAlert size={19} className="text-[var(--notumor)]" />
                <h2 className="font-display text-base font-bold text-ink-deep">
                  Prediction Confidence Distribution
                </h2>
              </div>
              <span className="font-mono text-[0.68rem] text-ink-muted">20 Bins</span>
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              &gt;82% of the 1,600 held-out test scans are evaluated with &gt;90% confidence,
              indicating strong model calibration. Amber bars mark the model's honest
              misclassification set.
            </p>
            <div className="mt-6">
              <ConfidenceHistogram />
            </div>
          </div>
        </div>
      </section>

      {/* ================= CHARTS GRID 3: ROC-AUC CURVES & HARDWARE GPU MONITOR ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="grid gap-8 lg:grid-cols-12">
          {/* ROC Curves (5 cols) */}
          <div className="liquid-card radius-card-lg p-6 lg:col-span-5">
            <div className="flex items-center justify-between border-b border-white/60 pb-3">
              <div className="flex items-center gap-2">
                <Activity size={19} className="text-[var(--cyan-liquid)]" />
                <h2 className="font-display text-base font-bold text-ink-deep">
                  Multi-Class ROC-AUC (One-vs-Rest)
                </h2>
              </div>
              <span className="font-mono text-[0.68rem] text-ink-muted">Micro AUC: 0.987</span>
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              Near-perfect separation across all four clinical categories.
            </p>
            <div className="mt-6 flex justify-center">
              <RocCurves />
            </div>
            <div className="mt-4">
              <RocLegend />
            </div>
          </div>

          {/* Model Comparison & GPU Console (7 cols) */}
          <div className="space-y-6 lg:col-span-7">
            {/* Multi-model comparison */}
            <div className="liquid-card radius-card-lg p-6">
              <div className="flex items-center justify-between border-b border-white/60 pb-3">
                <div className="flex items-center gap-2">
                  <Layers size={19} className="text-[var(--teal-soft)]" />
                  <h2 className="font-display text-base font-bold text-ink-deep">
                    Architecture Benchmarking on RTX 3050 (6GB)
                  </h2>
                </div>
                <span className="font-mono text-[0.68rem] text-ink-muted">3 Architectures</span>
              </div>
              <p className="mt-2 text-xs text-ink-muted">
                EfficientNet-B0 was selected for deployment due to superior macro-recall, half the
                latency, and a safe VRAM footprint.
              </p>
              <div className="mt-4">
                <ModelComparisonTable />
              </div>
            </div>

            {/* GPU Diagnostics */}
            <div className="liquid-card radius-card-lg p-6">
              <div className="flex items-center gap-2 border-b border-white/60 pb-3">
                <Server size={19} className="text-[var(--cyan-liquid)]" />
                <h2 className="font-display text-base font-bold text-ink-deep">
                  Hardware & VRAM Diagnostics
                </h2>
              </div>
              <div className="mt-4">
                <GpuMonitor />
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
