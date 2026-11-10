import { createFileRoute } from "@tanstack/react-router";
import { useState, useRef } from "react";
import {
  UploadCloud,
  FileImage,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sliders,
  Layers,
  Sparkles,
  Download,
  Contrast,
  TriangleAlert,
} from "lucide-react";
import scannerImg from "@/assets/mri_scanner_console.jpg";
import {
  runTumorInference,
  simulatePrediction,
  gradcamFocus,
  ApiError,
  type Prediction,
} from "@/lib/api";
import { CLASS_COLORS, formatPct } from "@/lib/metrics";

export const Route = createFileRoute("/detector")({
  component: DetectorPage,
});

/* Sample MRI scans generated as high-contrast canvas mockups for instant 1-click testing */
const SAMPLES = [
  { name: "Glioma Sample (Axial)", label: "Glioma", focus: { x: 38, y: 42 } },
  { name: "Meningioma Sample (Coronal)", label: "Meningioma", focus: { x: 70, y: 32 } },
  { name: "Pituitary Sample (Sagittal)", label: "Pituitary", focus: { x: 50, y: 64 } },
  { name: "Healthy Brain (No Tumor)", label: "No Tumor", focus: { x: 50, y: 50 } },
];

export function DetectorPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isSimulated, setIsSimulated] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  // Viewport Controls
  const [zoom, setZoom] = useState(1);
  const [isInverted, setIsInverted] = useState(false);
  const [gradcamOpacity, setGradcamOpacity] = useState(0.65);
  const [showGradcam, setShowGradcam] = useState(true);
  const [showContour, setShowContour] = useState(false);
  const [mouseCoords, setMouseCoords] = useState<{ x: number; y: number } | null>(null);
  const [focusPoint, setFocusPoint] = useState<{ x: number; y: number }>({ x: 42, y: 38 });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);

  // Create mock file from sample
  const handleSelectSample = (sample: (typeof SAMPLES)[number]) => {
    const canvas = document.createElement("canvas");
    canvas.width = 300;
    canvas.height = 300;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      // Cranial silhouette
      ctx.fillStyle = "#05070c";
      ctx.fillRect(0, 0, 300, 300);

      // Skull boundary
      ctx.fillStyle = "#162033";
      ctx.beginPath();
      ctx.ellipse(150, 150, 120, 135, 0, 0, Math.PI * 2);
      ctx.fill();

      // Brain cortex layers
      const grad = ctx.createRadialGradient(150, 150, 20, 150, 150, 115);
      grad.addColorStop(0, "#475569");
      grad.addColorStop(0.5, "#334155");
      grad.addColorStop(0.85, "#1e293b");
      grad.addColorStop(1, "#0f172a");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.ellipse(150, 150, 115, 130, 0, 0, Math.PI * 2);
      ctx.fill();

      // Sulci / Ventricle texture
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(150, 45);
      ctx.quadraticCurveTo(140, 150, 150, 255);
      ctx.stroke();

      // Ventricles
      ctx.fillStyle = "#080c14";
      ctx.beginPath();
      ctx.ellipse(135, 145, 14, 30, -0.15, 0, Math.PI * 2);
      ctx.ellipse(165, 145, 14, 30, 0.15, 0, Math.PI * 2);
      ctx.fill();

      // Anomaly spot if tumor sample
      if (sample.label !== "No Tumor") {
        const ax = (sample.focus.x / 100) * 300;
        const ay = (sample.focus.y / 100) * 300;
        const lesionGrad = ctx.createRadialGradient(ax, ay, 2, ax, ay, 26);
        lesionGrad.addColorStop(0, "#cbd5e1");
        lesionGrad.addColorStop(0.5, "#94a3b8");
        lesionGrad.addColorStop(1, "transparent");
        ctx.fillStyle = lesionGrad;
        ctx.beginPath();
        ctx.arc(ax, ay, 26, 0, Math.PI * 2);
        ctx.fill();
      }

      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], `${sample.label.toLowerCase()}_sample.png`, {
            type: "image/png",
          });
          loadScanFile(file, sample.focus);
        }
      });
    }
  };

  const loadScanFile = (file: File, customFocus?: { x: number; y: number }) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setPrediction(null);
    setScanError(null);
    setZoom(1);
    setFocusPoint(customFocus ?? gradcamFocus(file));
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      loadScanFile(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      loadScanFile(file);
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setIsSimulated(false);
    setScanError(null);

    try {
      const result = await runTumorInference(selectedFile);
      setPrediction(result);
    } catch (err) {
      if (err instanceof ApiError) {
        // The backend evaluated the scan and rejected it (e.g. color image
        // on a grayscale-only pipeline, corrupt file). Never fake a result —
        // surface the rejection to the user.
        setPrediction(null);
        setScanError(err.message);
      } else {
        // Backend unreachable: gracefully run deterministic demo simulation
        const sim = simulatePrediction(selectedFile);
        setPrediction(sim);
        setIsSimulated(true);
        setScanError("Engine unreachable — showing simulated demo result");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleMouseMoveViewport = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!viewportRef.current) return;
    const rect = viewportRef.current.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    setMouseCoords({ x, y });
  };

  const handleDownloadReport = () => {
    if (!prediction || !selectedFile) return;
    const reportData = {
      scan_file: selectedFile.name,
      file_size_bytes: selectedFile.size,
      timestamp: new Date().toISOString(),
      prediction: prediction.class_name,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      latency_ms: prediction.inference_time_ms,
      engine: "EfficientNet-B0 ONNX (ONNX Runtime CUDA)",
      is_simulated: isSimulated,
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `BrainTumor_Report_${prediction.class_name.toLowerCase()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getStatusColor = (className: string) => {
    return CLASS_COLORS[className] ?? "var(--cyan-liquid)";
  };

  return (
    <div className="space-y-12 md:space-y-16">
      {/* ================= PAGE HEADER (Standardized matching Performance & Early Stages) ================= */}
      <section className="container mx-auto px-4 max-w-6xl text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/60 px-3.5 py-1.5 shadow-sm backdrop-blur-md">
          <span className="h-2 w-2 rounded-full bg-[var(--cyan-liquid)] animate-pulse" />
          <span className="font-mono text-xs font-semibold text-ink-deep">
            High-Resolution MRI Scanner Studio · Diagnostic Console
          </span>
        </div>
        <h1 className="mt-4 font-display text-3xl font-extrabold tracking-tight text-ink-deep sm:text-4xl lg:text-5xl">
          AI Tumor Detector Studio
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-base text-ink-body md:text-lg leading-relaxed">
          Upload any cranial MRI slice (Axial, Coronal, Sagittal) to run instant GPU-accelerated
          classification with interactive Grad-CAM visual explainability.
        </p>
      </section>

      {/* ================= PRE-FLIGHT CHECKLIST & SCANNER SHOWCASE ================= */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="grid gap-6 lg:grid-cols-12 items-stretch">
          {/* Checklist (7 cols) */}
          <div className="liquid-card radius-card-md p-6 lg:col-span-7 flex flex-col justify-between">
            <div className="flex items-center gap-2.5 border-b border-white/60 pb-3">
              <Layers size={20} className="text-[var(--cyan-liquid)]" />
              <h2 className="font-display text-base sm:text-lg font-bold text-ink-deep">
                MRI Scanner Input Validation Checklist
              </h2>
            </div>
            <div className="mt-4 grid gap-4 sm:grid-cols-3">
              <div className="space-y-1.5">
                <span className="font-display text-sm font-bold text-ink-deep">1. Modalities</span>
                <ul className="space-y-1 text-xs text-ink-body leading-relaxed">
                  <li>• T1-weighted (+Gad)</li>
                  <li>• T2-weighted Axial</li>
                  <li>• FLAIR contrast</li>
                  <li>• JPEG, PNG, WebP</li>
                  <li>• Grayscale scans only</li>
                </ul>
              </div>
              <div className="space-y-1.5">
                <span className="font-display text-sm font-bold text-ink-deep">2. Orientations</span>
                <ul className="space-y-1 text-xs text-ink-body leading-relaxed">
                  <li>• Axial (Top-down)</li>
                  <li>• Coronal (Face-on)</li>
                  <li>• Sagittal (Profile)</li>
                  <li>• Auto-crop border</li>
                </ul>
              </div>
              <div className="space-y-1.5">
                <span className="font-display text-sm font-bold text-ink-deep">3. Standards</span>
                <ul className="space-y-1 text-xs text-ink-body leading-relaxed">
                  <li>• Min 224×224 px</li>
                  <li>• Low motion blur</li>
                  <li>• High contrast</li>
                  <li>• Size under 20MB</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Featured Scanner Image (5 cols) */}
          <div className="scanner-bay-frame liquid-card radius-card-md overflow-hidden lg:col-span-5 flex flex-col justify-between">
            <div className="glass-header-ribbon flex items-center justify-between border-b border-white/60 bg-white/40 px-4 py-2.5 text-xs">
              <div className="flex items-center gap-2 font-display text-xs sm:text-sm font-bold text-ink-deep">
                <span className="pulse-indicator h-2 w-2 rounded-full bg-[var(--teal-soft)] animate-pulse" />
                <span>3.0T Neuroimaging Suite</span>
              </div>
              <span className="font-mono text-[0.7rem] font-semibold text-ink-muted">ONNX GPU Ready</span>
            </div>
            <img
              src={scannerImg}
              alt="Advanced 3.0 Tesla Cranial MRI Scanner Console"
              className="aspect-video max-h-[190px] w-full object-cover"
              loading="lazy"
              width={540}
              height={300}
            />
            <div className="flex items-center justify-around border-t border-white/60 bg-white/50 px-3 py-2 text-xs">
              <span className="text-ink-body">Field: <strong className="font-bold text-ink-deep">3.0 Tesla</strong></span>
              <span className="text-ink-body">Slice: <strong className="font-bold text-ink-deep">1.0 mm 3D</strong></span>
              <span className="text-ink-body">Latency: <strong className="font-bold text-ink-deep">11.4 ms</strong></span>
            </div>
          </div>
        </div>
      </section>

      {/* ================= INTERACTIVE SCANNER CONSOLE ================= */}
      <section className="container mx-auto px-4 max-w-6xl">
        <div className="grid gap-6 lg:grid-cols-12 items-start">
          {/* LEFT PANEL: Viewport & Upload Console (7 cols) */}
          <div className="space-y-3.5 lg:col-span-7">
            {/* Quick Sample Selector */}
            <div className="flex flex-wrap items-center gap-2 bg-white/60 px-4 py-2 rounded-2xl border border-white/70 shadow-sm">
              <span className="font-display text-xs sm:text-sm font-bold text-ink-deep">
                Try Clinical Samples:
              </span>
              {SAMPLES.map((sample) => (
                <button
                  key={sample.name}
                  onClick={() => handleSelectSample(sample)}
                  className="rounded-full border border-white/80 bg-white/90 px-3 py-1 font-mono text-xs font-semibold text-ink-body shadow-xs transition hover:bg-white hover:text-[var(--cyan-liquid)] hover:shadow-sm"
                >
                  {sample.label}
                </button>
              ))}
            </div>

            {/* Viewport Card */}
            <div className="liquid-card radius-card-md overflow-hidden p-5">
              {/* Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/60 pb-3 text-xs">
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => setZoom((z) => Math.min(z + 0.25, 3))}
                    className="rounded-lg p-1.5 hover:bg-white/80 text-ink-deep transition"
                    title="Zoom in"
                  >
                    <ZoomIn size={16} />
                  </button>
                  <button
                    onClick={() => setZoom((z) => Math.max(z - 0.25, 0.75))}
                    className="rounded-lg p-1.5 hover:bg-white/80 text-ink-deep transition"
                    title="Zoom out"
                  >
                    <ZoomOut size={16} />
                  </button>
                  <button
                    onClick={() => setZoom(1)}
                    className="rounded-lg p-1.5 hover:bg-white/80 text-ink-deep transition"
                    title="Reset zoom"
                  >
                    <RotateCcw size={15} />
                  </button>
                  <span className="font-mono text-xs font-bold text-ink-muted ml-1">
                    {(zoom * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsInverted(!isInverted)}
                    className={`rounded-lg px-3 py-1 text-xs font-semibold transition ${
                      isInverted
                        ? "bg-[var(--cyan-liquid)] text-white shadow-sm"
                        : "bg-white/70 text-ink-deep hover:bg-white/90 border border-white/80"
                    }`}
                  >
                    <Contrast size={13} className="inline mr-1.5" />
                    Invert Grayscale
                  </button>

                  <button
                    onClick={() => setShowContour(!showContour)}
                    className={`rounded-lg px-3 py-1 text-xs font-semibold transition ${
                      showContour
                        ? "bg-[var(--teal-soft)] text-white shadow-sm"
                        : "bg-white/70 text-ink-deep hover:bg-white/90 border border-white/80"
                    }`}
                  >
                    Skull Contour
                  </button>
                </div>
              </div>

              {/* Viewport Screen */}
              <div
                ref={viewportRef}
                onMouseMove={handleMouseMoveViewport}
                onMouseLeave={() => setMouseCoords(null)}
                className="relative mt-3.5 flex min-h-[300px] max-h-[360px] items-center justify-center overflow-hidden rounded-2xl bg-[#070b14] shadow-inner select-none"
              >
                {previewUrl ? (
                  <div
                    className="relative transition-transform duration-100 ease-out"
                    style={{
                      transform: `scale(${zoom})`,
                      filter: isInverted ? "invert(1) hue-rotate(180deg)" : "none",
                    }}
                  >
                    <img
                      src={previewUrl}
                      alt="Brain MRI scan under analysis"
                      className="max-h-[300px] max-w-full rounded-lg object-contain"
                    />

                    {/* Skull contour overlay simulation */}
                    {showContour && (
                      <div className="pointer-events-none absolute inset-0 rounded-full border-2 border-dashed border-[var(--teal-soft)]/60" />
                    )}

                    {/* Grad-CAM Heatmap Overlay */}
                    {showGradcam && prediction && prediction.class_name !== "No Tumor" && (
                      <div
                        className="pointer-events-none absolute inset-0 overflow-hidden rounded-lg transition-opacity"
                        style={{ opacity: gradcamOpacity }}
                      >
                        <div
                          className="absolute h-32 w-32 -translate-x-1/2 -translate-y-1/2 rounded-full"
                          style={{
                            left: `${focusPoint.x}%`,
                            top: `${focusPoint.y}%`,
                            background:
                              "radial-gradient(circle, rgba(220,38,38,0.95) 0%, rgba(245,158,11,0.75) 35%, rgba(56,189,248,0.4) 65%, transparent 100%)",
                            filter: "blur(13px)",
                            mixBlendMode: "screen",
                          }}
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  /* Empty Dropzone State */
                  <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className="flex flex-col items-center justify-center p-8 text-center cursor-pointer transition hover:opacity-90"
                  >
                    <div className="mb-3 flex h-13 w-13 items-center justify-center rounded-2xl bg-white/10 text-[var(--cyan-liquid)] backdrop-blur-md">
                      <UploadCloud size={26} />
                    </div>
                    <p className="font-display text-sm font-bold text-white">
                      Drag & Drop Cranial MRI Scan Here or Click to Browse
                    </p>
                    <p className="mt-1 text-xs text-slate-400">
                      Supports axial, coronal, and sagittal DICOM/JPEG/PNG slices
                    </p>
                    <span className="mt-3.5 rounded-full border border-white/20 bg-white/10 px-4 py-1.5 text-xs font-semibold text-white backdrop-blur-sm">
                      Select File from Workstation
                    </span>
                  </div>
                )}

                {/* Real-time Crosshair Tracking HUD */}
                {mouseCoords && (
                  <div className="pointer-events-none absolute bottom-3 right-3 rounded-md bg-black/75 px-2.5 py-1 font-mono text-[0.68rem] text-emerald-400 backdrop-blur-sm">
                    X: {mouseCoords.x}px · Y: {mouseCoords.y}px
                  </div>
                )}
              </div>

              {/* Hidden File Input */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileUpload}
                className="hidden"
              />

              {/* Grad-CAM Slider & Run Action Bar */}
              <div className="mt-3.5 space-y-3">
                {prediction && prediction.class_name !== "No Tumor" && (
                  <div className="flex items-center gap-3 rounded-xl bg-white/70 px-3.5 py-2.5 text-xs">
                    <Sliders size={16} className="text-[var(--cyan-liquid)]" />
                    <span className="font-display text-xs font-bold text-ink-deep">Grad-CAM Opacity:</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={gradcamOpacity}
                      onChange={(e) => setGradcamOpacity(parseFloat(e.target.value))}
                      className="flex-1 accent-[var(--cyan-liquid)] cursor-pointer h-2"
                    />
                    <span className="w-10 text-right font-mono text-xs font-bold text-ink-deep">
                      {(gradcamOpacity * 100).toFixed(0)}%
                    </span>
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-2.5">
                  <button
                    onClick={handleRunInference}
                    disabled={!selectedFile || isProcessing}
                    className="btn-liquid-primary flex-1 inline-flex items-center justify-center gap-2 !py-3 text-sm font-bold disabled:opacity-50"
                  >
                    {isProcessing ? (
                      <>
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        <span>Evaluating Neural Features…</span>
                      </>
                    ) : (
                      <>
                        <Sparkles size={16} />
                        <span>Run Deep Neural Diagnostic Scan</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="btn-liquid-secondary !py-3 !px-4 text-xs font-bold"
                  >
                    Change File
                  </button>
                </div>

                {/* Scan rejection / engine warning banner */}
                {scanError && (
                  <div
                    role="alert"
                    className={`flex items-start gap-2.5 rounded-xl px-3.5 py-2.5 text-xs font-medium ${
                      isSimulated
                        ? "border border-amber-200 bg-amber-50 text-amber-700"
                        : "border border-red-200 bg-red-50 text-red-700"
                    }`}
                  >
                    <TriangleAlert size={15} className="mt-0.5 shrink-0" />
                    <span>{scanError}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* RIGHT PANEL: Real-Time Diagnostic Results Card (5 cols) */}
          <div className="lg:col-span-5">
            <div className="liquid-card radius-card-md space-y-5 p-6">
              <div className="flex items-center justify-between border-b border-white/60 pb-3">
                <h2 className="font-display text-base font-bold text-ink-deep">
                  Diagnostic Evaluation Report
                </h2>
                {isSimulated && (
                  <span className="rounded-full bg-amber-500/15 px-2.5 py-0.5 text-[0.68rem] font-bold text-amber-600">
                    DEMO SIMULATION
                  </span>
                )}
              </div>

              {prediction ? (
                <>
                  {/* Master Banner */}
                  <div
                    className="rounded-2xl p-4 text-white shadow-md transition-all"
                    style={{
                      background: `linear-gradient(135deg, ${getStatusColor(
                        prediction.class_name
                      )}, #0a1120)`,
                    }}
                  >
                    <span className="font-mono text-xs uppercase tracking-wider font-semibold opacity-90">
                      Primary Classification
                    </span>
                    <h3 className="mt-1 font-display text-2xl font-black">
                      {prediction.class_name.toUpperCase()}
                      {prediction.class_name !== "No Tumor" ? " DETECTED" : " CONFIRMED"}
                    </h3>
                    <p className="mt-1 text-xs sm:text-sm leading-relaxed opacity-95">
                      {prediction.class_name === "No Tumor"
                        ? "Normal cranial architecture. No pathological tumor mass observed."
                        : `Localized abnormal tissue pattern identified with ${(
                            prediction.confidence * 100
                          ).toFixed(1)}% certainty.`}
                    </p>
                  </div>

                  {/* Circular Liquid Confidence Meter */}
                  <div className="flex flex-col items-center justify-center p-2 text-center">
                    <div className="relative flex h-32 w-32 items-center justify-center rounded-full border-4 border-white bg-white/70 shadow-md backdrop-blur-md">
                      <div
                        className="absolute inset-1 rounded-full opacity-20"
                        style={{ background: getStatusColor(prediction.class_name) }}
                      />
                      <div className="relative z-10 flex flex-col items-center">
                        <span className="font-mono text-3xl font-extrabold text-ink-deep">
                          {formatPct(prediction.confidence, 1)}
                        </span>
                        <span className="text-xs font-semibold text-ink-muted">
                          Confidence Score
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* 4-Class Probability Spectrum */}
                  <div className="space-y-2.5">
                    <span className="font-display text-xs sm:text-sm font-bold text-ink-deep block">
                      Multi-Class Probability Distribution
                    </span>
                    <div className="space-y-2">
                      {Object.entries(prediction.probabilities).map(([cls, prob]) => {
                        const isWinner = cls === prediction.class_name;
                        return (
                          <div key={cls} className="space-y-1">
                            <div className="flex justify-between text-xs sm:text-sm">
                              <span
                                className={`font-medium ${
                                  isWinner ? "font-bold text-ink-deep" : "text-ink-muted"
                                }`}
                              >
                                {cls}
                              </span>
                              <span className="font-mono font-bold text-ink-deep">
                                {formatPct(prob, 1)}
                              </span>
                            </div>
                            <div className="liquid-track !h-2.5">
                              <div
                                className="liquid-fill"
                                style={{
                                  width: `${prob * 100}%`,
                                  background: getStatusColor(cls),
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Inference Diagnostics Capsule */}
                  <div className="rounded-xl border border-white/70 bg-white/50 p-3.5 text-xs space-y-1.5 font-mono text-ink-body">
                    <div className="flex justify-between">
                      <span className="text-ink-muted">Inference Latency:</span>
                      <strong className="font-bold text-ink-deep">{prediction.inference_time_ms.toFixed(1)} ms</strong>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-ink-muted">Model Backbone:</span>
                      <span className="font-bold text-ink-deep">EfficientNet-B0 ONNX</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-ink-muted">Execution Device:</span>
                      <span className="font-bold text-ink-deep">NVIDIA RTX 3050 (CUDA)</span>
                    </div>
                  </div>

                  {/* Action Bar */}
                  <div className="flex flex-wrap gap-2.5 pt-1">
                    <button
                      onClick={handleDownloadReport}
                      className="btn-liquid-secondary flex-1 inline-flex items-center justify-center gap-2 !py-2.5 text-xs font-bold"
                    >
                      <Download size={15} />
                      <span>Download JSON Report</span>
                    </button>
                    <button
                      onClick={() => {
                        setSelectedFile(null);
                        setPreviewUrl(null);
                        setPrediction(null);
                      }}
                      className="btn-liquid-secondary !py-2.5 !px-4 text-xs font-bold"
                    >
                      Reset View
                    </button>
                  </div>
                </>
              ) : (
                /* Empty state */
                <div className="flex flex-col items-center justify-center py-14 text-center text-ink-muted">
                  <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-white/60">
                    <FileImage size={24} className="text-[var(--cyan-liquid)]" />
                  </div>
                  <h3 className="font-display text-sm font-bold text-ink-deep">
                    Awaiting MRI Input
                  </h3>
                  <p className="mt-1 max-w-[220px] text-xs leading-relaxed">
                    Choose a clinical sample on the left or upload your own scan to view real-time AI
                    diagnostics.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
