/**
 * Custom SVG chart engines for the Performance & Analytics page,
 * themed with the liquid glass design tokens.
 */
import {
  CLASSES,
  CLASS_COLORS,
  CONFUSION_MATRIX,
  TEST_SAMPLES,
  PER_CLASS,
  MACRO,
  TRAINING_CURVES,
  CORRECT_CONFIDENCE_BINS,
  AMBIGUOUS_CONFIDENCE_BINS,
  ROC_CURVES,
  MODEL_COMPARISON,
  formatPct,
  BENCHMARK,
} from "@/lib/metrics";

/* ================= Confusion Matrix ================= */

export function ConfusionMatrixChart() {
  const cellW = 95;
  const cellH = 78;
  const x0 = 118;
  const y0 = 66;

  return (
    <svg viewBox="0 0 520 398" className="w-full" role="img" aria-label="4x4 confusion matrix heatmap">
      {CLASSES.map((pred, c) => (
        <text
          key={pred}
          x={x0 + c * cellW + cellW / 2}
          y={y0 - 12}
          textAnchor="middle"
          fontSize={11.5}
          fontWeight={700}
          fill="var(--ink-body)"
          fontFamily="var(--font-mono)"
        >
          {pred}
        </text>
      ))}
      {CLASSES.map((trueCls, r) => (
        <g key={trueCls}>
          <text
            x={x0 - 10}
            y={y0 + r * cellH + cellH / 2 + 4}
            textAnchor="end"
            fontSize={11.5}
            fontWeight={700}
            fill="var(--ink-body)"
            fontFamily="var(--font-mono)"
          >
            {trueCls}
          </text>
          {CLASSES.map((predCls, c) => {
            const count = CONFUSION_MATRIX[r]?.[c] ?? 0;
            const total = TEST_SAMPLES[r] ?? 1;
            const pct = count / total;
            const alpha = 0.08 + pct * 0.82;
            const onDiagonal = r === c;
            return (
              <g key={predCls}>
                <rect
                  className="matrix-cell"
                  x={x0 + c * cellW + 3}
                  y={y0 + r * cellH + 3}
                  width={cellW - 6}
                  height={cellH - 6}
                  rx={12}
                  fill={onDiagonal ? CLASS_COLORS[trueCls] : "var(--cyan-liquid)"}
                  fillOpacity={onDiagonal ? alpha : alpha * 0.55}
                  stroke="rgba(255,255,255,0.7)"
                >
                  <title>{`True ${trueCls} → Predicted ${predCls}: ${count} scans (${formatPct(pct, 1)})`}</title>
                </rect>
                <text
                  x={x0 + c * cellW + cellW / 2}
                  y={y0 + r * cellH + cellH / 2 - 2}
                  textAnchor="middle"
                  fontSize={15}
                  fontWeight={700}
                  fill="var(--ink-deep)"
                  pointerEvents="none"
                >
                  {count}
                </text>
                <text
                  x={x0 + c * cellW + cellW / 2}
                  y={y0 + r * cellH + cellH / 2 + 14}
                  textAnchor="middle"
                  fontSize={9.5}
                  fontWeight={500}
                  fill="var(--ink-muted)"
                  pointerEvents="none"
                >
                  {formatPct(pct, 1)}
                </text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

/* ================= Per-class grouped bars ================= */

export function MetricsBars() {
  const plotX0 = 44;
  const plotY0 = 246;
  const plotH = 206; // 0.80 -> 1.00 scale
  const groupW = 128;

  const yFor = (v: number) => plotY0 - ((v - 0.8) / 0.2) * plotH;

  return (
    <svg viewBox="0 0 560 292" className="w-full" role="img" aria-label="Per-class precision, recall and F1 bars">
      {[0.8, 0.85, 0.9, 0.95, 1.0].map((t) => (
        <g key={t}>
          <line x1={plotX0} x2={552} y1={yFor(t)} y2={yFor(t)} stroke="rgba(2,132,199,0.14)" strokeDasharray="3 4" />
          <text x={plotX0 - 6} y={yFor(t) + 3.5} textAnchor="end" fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-mono)">
            {(t * 100).toFixed(1)}
          </text>
        </g>
      ))}
      {PER_CLASS.map((cls, gi) => (
        <g key={cls.name}>
          {(
            [
              ["Precision", cls.precision, "var(--cyan-liquid)"],
              ["Recall", cls.recall, CLASS_COLORS[cls.name]],
              ["F1-Score", cls.f1, "var(--teal-soft)"],
            ] as const
          ).map(([metric, value, color], bi) => {
            const x = plotX0 + gi * groupW + 10 + bi * 26;
            const y = yFor(value);
            return (
              <g key={metric}>
                <rect x={x} y={y} width={20} height={plotY0 - y} rx={7} fill={color} opacity={0.9}>
                  <title>{`${cls.name} — ${metric}: ${formatPct(value, 1)}`}</title>
                </rect>
              </g>
            );
          })}
          <text x={plotX0 + gi * groupW + 42} y={272} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-body)">
            {cls.name}
          </text>
        </g>
      ))}
    </svg>
  );
}

/* ================= Radar (spider) chart ================= */

export function MetricsRadar() {
  const cx = 160;
  const cy = 148;
  const R = 108;
  const axes = ["Precision", "Recall", "F1-Score"];
  const angle = (i: number) => (Math.PI * 2 * i) / 3 - Math.PI / 2;
  const point = (i: number, v: number) => {
    const a = angle(i);
    return `${cx + Math.cos(a) * R * ((v - 0.8) / 0.2)},${cy + Math.sin(a) * R * ((v - 0.8) / 0.2)}`;
  };

  return (
    <svg viewBox="0 0 320 300" className="w-full" role="img" aria-label="Per-class metrics spider radar">
      {[0.8, 0.9, 1.0].map((ring) => (
        <polygon
          key={ring}
          points={[0, 1, 2].map((i) => point(i, ring)).join(" ")}
          fill="none"
          stroke="rgba(2,132,199,0.18)"
        />
      ))}
      {[0, 1, 2].map((i) => {
        const a = angle(i);
        return (
          <g key={i}>
            <line x1={cx} y1={cy} x2={cx + Math.cos(a) * R} y2={cy + Math.sin(a) * R} stroke="rgba(2,132,199,0.18)" />
            <text
              x={cx + Math.cos(a) * (R + 16)}
              y={cy + Math.sin(a) * (R + 16) + 4}
              textAnchor="middle"
              fontSize={10.5}
              fontWeight={700}
              fill="var(--ink-body)"
            >
              {axes[i]}
            </text>
          </g>
        );
      })}
      {PER_CLASS.map((cls) => (
        <polygon
          key={cls.name}
          points={[cls.precision, cls.recall, cls.f1].map((v, i) => point(i, v)).join(" ")}
          fill={CLASS_COLORS[cls.name]}
          fillOpacity={0.16}
          stroke={CLASS_COLORS[cls.name]}
          strokeWidth={2}
        >
          <title>{`${cls.name}: P ${formatPct(cls.precision, 1)} · R ${formatPct(cls.recall, 1)} · F1 ${formatPct(cls.f1, 1)}`}</title>
        </polygon>
      ))}
    </svg>
  );
}

export function RadarLegend() {
  return (
    <div className="flex flex-wrap justify-center gap-3">
      {PER_CLASS.map((cls) => (
        <span key={cls.name} className="flex items-center gap-1.5 text-[0.78rem] font-semibold text-ink-body">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: CLASS_COLORS[cls.name] }} />
          {cls.name}
        </span>
      ))}
      <span className="text-[0.78rem] font-semibold text-ink-muted">
        Macro: P {formatPct(MACRO.precision, 2)} · R {formatPct(MACRO.recall, 2)} · F1 {formatPct(MACRO.f1, 2)}
      </span>
    </div>
  );
}

/* ================= Two-stage training curves ================= */

export function TrainingCurves() {
  const W = 660;
  const plotX0 = 56;
  const plotX1 = 612;
  const plotY0 = 268;
  const plotY1 = 30;
  const n = TRAINING_CURVES.trainLoss.length;
  const xFor = (i: number) => plotX0 + ((i - 1) / (n - 1)) * (plotX1 - plotX0);
  const yLoss = (v: number) => plotY0 - (v / 1.5) * (plotY0 - plotY1);
  const yRecall = (v: number) => plotY0 - ((v - 0.6) / 0.4) * (plotY0 - plotY1);

  const line = (vals: readonly number[], yF: (v: number) => number) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${xFor(i + 1).toFixed(1)},${yF(v).toFixed(1)}`).join(" ");

  const splitX = xFor(TRAINING_CURVES.stage1End + 0.5);

  return (
    <svg viewBox={`0 0 ${W} 320`} className="w-full" role="img" aria-label="Two-stage training and loss curves">
      {/* grid */}
      {[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5].map((t) => (
        <g key={t}>
          <line x1={plotX0} x2={plotX1} y1={yLoss(t)} y2={yLoss(t)} stroke="rgba(2,132,199,0.12)" strokeDasharray="3 4" />
          <text x={plotX0 - 6} y={yLoss(t) + 3.5} textAnchor="end" fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-mono)">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {[0.6, 0.7, 0.8, 0.9, 1.0].map((t) => (
        <text key={t} x={plotX1 + 6} y={yRecall(t) + 3.5} fontSize={9.5} fill="var(--teal-soft)" fontFamily="var(--font-mono)">
          {(t * 100).toFixed(0)}%
        </text>
      ))}
      {/* stage splitter */}
      <line x1={splitX} x2={splitX} y1={plotY1 - 6} y2={plotY0} stroke="var(--pituitary)" strokeWidth={1.5} strokeDasharray="5 4" />
      <text x={splitX - 8} y={plotY1 + 2} textAnchor="end" fontSize={9.5} fontWeight={700} fill="var(--pituitary)">
        Stage 1 · Head Only
      </text>
      <text x={splitX + 8} y={plotY1 + 2} fontSize={9.5} fontWeight={700} fill="var(--pituitary)">
        Stage 2 · Full Fine-Tune (cosine LR)
      </text>
      {/* curves */}
      <path d={line(TRAINING_CURVES.trainLoss, yLoss)} fill="none" stroke="var(--cyan-liquid)" strokeWidth={2.5} strokeLinecap="round" />
      <path d={line(TRAINING_CURVES.valLoss, yLoss)} fill="none" stroke="var(--pituitary)" strokeWidth={2} strokeDasharray="6 4" />
      <path d={line(TRAINING_CURVES.valRecall, yRecall)} fill="none" stroke="var(--teal-soft)" strokeWidth={2.5} strokeLinecap="round" />
      {/* hover points */}
      {TRAINING_CURVES.valRecall.map((r, i) => (
        <circle key={i} cx={xFor(i + 1)} cy={yRecall(r)} r={7} fill="transparent">
          <title>{`Epoch ${i + 1} · LR ${TRAINING_CURVES.learningRates[i] ?? ""} · Train loss ${TRAINING_CURVES.trainLoss[i]?.toFixed(2) ?? ""} · Val loss ${TRAINING_CURVES.valLoss[i]?.toFixed(2) ?? ""} · Val recall ${formatPct(r, 2)}`}</title>
        </circle>
      ))}
      {/* axes labels */}
      {[1, 5, 10, 15, 21].map((e) => (
        <text key={e} x={xFor(e)} y={plotY0 + 16} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-mono)">
          {e}
        </text>
      ))}
      <text x={(plotX0 + plotX1) / 2} y={308} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="var(--ink-muted)">
        Epoch (1–21)
      </text>
      <g transform="translate(66, 296)" fontSize={9.5} fill="var(--ink-muted)">
        <rect width={10} height={4} rx={2} y={-4} fill="var(--cyan-liquid)" />
        <text x={15} y={0}>Train loss</text>
        <rect width={10} height={4} rx={2} x={80} y={-4} fill="var(--pituitary)" />
        <text x={95} y={0}>Val loss</text>
        <rect width={10} height={4} rx={2} x={152} y={-4} fill="var(--teal-soft)" />
        <text x={167} y={0}>Val macro-recall (right axis)</text>
      </g>
    </svg>
  );
}

/* ================= Confidence histogram ================= */

export function ConfidenceHistogram() {
  const bins = CORRECT_CONFIDENCE_BINS.length;
  const x0 = 56;
  const binW = 26;
  const gap = 3;
  const plotY0 = 258;
  const maxCorrect = 1200;
  const amberMax = 14;

  return (
    <svg viewBox="0 0 620 300" className="w-full" role="img" aria-label="Prediction confidence distribution histogram">
      <line x1={x0 - 10} x2={x0 + bins * (binW + gap)} y1={plotY0} y2={plotY0} stroke="rgba(2,132,199,0.3)" />
      {CORRECT_CONFIDENCE_BINS.map((count, i) => {
        const h = (count / maxCorrect) * 218;
        const x = x0 + i * (binW + gap);
        return (
          <g key={i}>
            <rect x={x} y={plotY0 - h} width={binW} height={h} rx={5} fill="var(--notumor)" opacity={0.88}>
              <title>{`Correct predictions with confidence ${(i / 20).toFixed(2)}–${((i + 1) / 20).toFixed(2)}: ${count} scans`}</title>
            </rect>
            {(AMBIGUOUS_CONFIDENCE_BINS[i] ?? 0) > 0 && (
              <rect
                x={x + binW / 4}
                y={plotY0 - ((AMBIGUOUS_CONFIDENCE_BINS[i] ?? 0) / amberMax) * 70}
                width={binW / 2}
                height={((AMBIGUOUS_CONFIDENCE_BINS[i] ?? 0) / amberMax) * 70}
                rx={4}
                fill="var(--meningioma)"
                opacity={0.75}
              >
                <title>{`Misclassified scan with confidence ${(i / 20).toFixed(2)}–${((i + 1) / 20).toFixed(2)}: ${AMBIGUOUS_CONFIDENCE_BINS[i] ?? 0}`}</title>
              </rect>
            )}
          </g>
        );
      })}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <text key={t} x={x0 + t * bins * (binW + gap)} y={plotY0 + 16} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-mono)">
          {t.toFixed(2)}
        </text>
      ))}
      <text x={(x0 * 2 + bins * (binW + gap)) / 2} y={294} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="var(--ink-muted)">
        Prediction confidence (0.0 – 1.0)
      </text>
      <g transform="translate(370, 40)" fontSize={10} fill="var(--ink-muted)">
        <rect width={10} height={10} rx={3} fill="var(--notumor)" opacity={0.88} />
        <text x={16} y={9}>Correct predictions</text>
        <rect width={10} height={10} rx={3} y={18} fill="var(--meningioma)" opacity={0.75} />
        <text x={16} y={27}>Misclassified</text>
      </g>
    </svg>
  );
}

/* ================= ROC curves ================= */

export function RocCurves() {
  const p0 = 52;
  const p1 = 388;
  const sx = (f: number) => p0 + f * (p1 - p0);
  const sy = (t: number) => p1 - t * (p1 - p0);

  return (
    <svg viewBox="0 0 420 440" className="w-full" role="img" aria-label="Multi-class ROC curves">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={p0} x2={p1} y1={sy(t)} y2={sy(t)} stroke="rgba(2,132,199,0.12)" />
          <line x1={sx(t)} x2={sx(t)} y1={p1} y2={p0} stroke="rgba(2,132,199,0.12)" />
          <text x={p0 - 8} y={sy(t) + 4} textAnchor="end" fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-mono)">
            {t.toFixed(2)}
          </text>
          <text x={sx(t)} y={p1 + 16} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)" fontFamily="var(--font-mono)">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      <line x1={p0} y1={p1} x2={p1} y2={p0} stroke="rgba(59,74,98,0.4)" strokeDasharray="5 5" />
      <text x={(p0 + p1) / 2} y={412} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="var(--ink-muted)">
        False Positive Rate
      </text>
      <text x={16} y={(p0 + p1) / 2} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="var(--ink-muted)" transform={`rotate(-90 16 ${(p0 + p1) / 2})`}>
        True Positive Rate
      </text>
      {ROC_CURVES.map((curve) => (
        <polyline
          key={curve.name}
          points={curve.points.map(([f, t]) => `${sx(f).toFixed(1)},${sy(t).toFixed(1)}`).join(" ")}
          fill="none"
          stroke={
            curve.name === "Micro-Average"
              ? "var(--ink-deep)"
              : CLASS_COLORS[curve.name]
          }
          strokeWidth={2.4}
          strokeDasharray={curve.name === "Micro-Average" ? "7 5" : undefined}
          strokeLinecap="round"
        >
          <title>{`${curve.name} — AUC ${curve.auc.toFixed(3)}`}</title>
        </polyline>
      ))}
    </svg>
  );
}

export function RocLegend() {
  return (
    <div className="flex flex-wrap justify-center gap-x-5 gap-y-2">
      {ROC_CURVES.map((c) => (
        <span key={c.name} className="flex items-center gap-1.5 text-[0.78rem] font-semibold text-ink-body">
          <span
            className="h-2.5 w-2.5 rounded-full"
            style={{ background: c.name === "Micro-Average" ? "var(--ink-deep)" : CLASS_COLORS[c.name] }}
          />
          {c.name} — AUC {c.auc.toFixed(3)}
        </span>
      ))}
    </div>
  );
}

/* ================= Multi-model comparison ================= */

export function ModelComparisonTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] border-separate border-spacing-y-2 text-left">
        <thead>
          <tr className="text-[0.72rem] uppercase tracking-wider text-ink-muted">
            <th className="px-4 py-2 font-semibold">Architecture</th>
            <th className="px-4 py-2 font-semibold">Params</th>
            <th className="px-4 py-2 font-semibold">Batch</th>
            <th className="px-4 py-2 font-semibold">Test Macro Recall</th>
            <th className="px-4 py-2 font-semibold">Test Accuracy</th>
            <th className="px-4 py-2 font-semibold">Latency</th>
            <th className="px-4 py-2 font-semibold">VRAM</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_COMPARISON.map((m) => (
            <tr
              key={m.name}
              className={`liquid-card-subtle radius-card-md ${m.deployed ? "ring-2 ring-[var(--teal-soft)]/50" : ""}`}
            >
              <td className="rounded-l-2xl px-4 py-3 font-display text-[0.9rem] font-bold text-ink-deep">
                {m.name}
                {m.deployed && (
                  <span className="ml-2 rounded-full bg-[var(--teal-soft)]/15 px-2 py-0.5 text-[0.65rem] font-bold text-[var(--teal-soft)]">
                    DEPLOYED
                  </span>
                )}
              </td>
              <td className="px-4 py-3 font-mono text-sm text-ink-body">{m.params}</td>
              <td className="px-4 py-3 font-mono text-sm text-ink-body">
                {m.batch}
                {"gradAccum" in m ? ` (GA ${m.gradAccum})` : ""}
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="liquid-track w-20 !h-2">
                    <div className="liquid-fill" style={{ width: `${m.macroRecall * 100}%` }} />
                  </div>
                  <span className="font-mono text-sm font-bold text-ink-deep">{formatPct(m.macroRecall)}</span>
                </div>
              </td>
              <td className="px-4 py-3 font-mono text-sm text-ink-body">{formatPct(m.accuracy)}</td>
              <td className="px-4 py-3 font-mono text-sm text-ink-body">{m.latencyMs.toFixed(1)} ms</td>
              <td className="rounded-r-2xl px-4 py-3 font-mono text-sm text-ink-body">{m.vramGb.toFixed(1)} GB</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ================= GPU monitor ================= */

export function GpuMonitor() {
  const pct = (BENCHMARK.vramUsedGb / BENCHMARK.vramTotalGb) * 100;
  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <span className="font-display text-sm font-bold text-ink-deep">VRAM Allocation</span>
        <span className="font-mono text-sm font-bold text-ink-deep">
          {BENCHMARK.vramUsedGb.toFixed(1)} GB / {BENCHMARK.vramTotalGb.toFixed(1)} GB
          <span className="ml-1 text-[0.72rem] font-medium text-ink-muted">({pct.toFixed(0)}% utilized)</span>
        </span>
      </div>
      <div className="liquid-track">
        <div className="liquid-fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="text-[0.8rem] text-ink-muted">1.2 GB safety margin retained.</p>
      <div className="flex flex-wrap gap-2">
        {["NVIDIA RTX 3050 Laptop GPU · Ampere · 2048 CUDA cores", "FP16 AMP enabled", "Channels-Last memory format", "cuDNN benchmark active"].map(
          (t) => (
            <span key={t} className="glass-pill !text-[0.72rem]">
              {t}
            </span>
          )
        )}
      </div>
    </div>
  );
}
