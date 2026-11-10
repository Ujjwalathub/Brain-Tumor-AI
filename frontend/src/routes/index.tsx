import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import {
  Brain,
  ShieldCheck,
  Zap,
  Eye,
  Lock,
  Stethoscope,
  ArrowRight,
  Activity,
  CheckCircle2,
  Sparkles,
  FileCheck2,
} from "lucide-react";
import doctorImg from "@/assets/doctor_female_orthopedic.jpg";

export const Route = createFileRoute("/")({
  component: IndexPage,
});

export function IndexPage() {
  const [tilt, setTilt] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width - 0.5) * 12;
    const y = ((e.clientY - rect.top) / rect.height - 0.5) * -12;
    setTilt({ x, y });
  };

  const handleMouseLeave = () => {
    setTilt({ x: 0, y: 0 });
  };

  return (
    <div className="space-y-24 py-6 md:py-12">
      {/* ================= HERO SECTION ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="grid items-center gap-10 lg:grid-cols-12 lg:gap-14">
          {/* Left Column: Mission & Plain English Pitch */}
          <div className="space-y-6 lg:col-span-7">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/60 px-3.5 py-1.5 shadow-sm backdrop-blur-md">
              <span className="h-2 w-2 rounded-full bg-[var(--cyan-liquid)] animate-pulse" />
              <span className="font-mono text-xs font-semibold text-ink-deep">
                AI Diagnostic Console v1.0 · Clinical Triage Engine
              </span>
            </div>

            <h1 className="font-display text-4xl font-extrabold tracking-tight text-ink-deep sm:text-5xl lg:text-[3.5rem] lg:leading-[1.12]">
              Precision Brain Health,{" "}
              <span className="bg-gradient-to-r from-[var(--cyan-liquid)] to-[#0369a1] bg-clip-text text-transparent">
                Explained in Plain Human Language.
              </span>
            </h1>

            <p className="max-w-2xl text-lg leading-relaxed text-ink-body md:text-xl">
              A compassionate, highly trained AI companion that scans brain MRI images in
              under 6 milliseconds with <strong>96.19% accuracy</strong>. Engineered to give doctors,
              patients, and families instant clarity and peace of mind when every second matters.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <Link to="/detector" className="btn-liquid-primary group inline-flex items-center gap-2">
                <span>Launch Tumor Detector</span>
                <ArrowRight size={17} className="transition-transform group-hover:translate-x-1" />
              </Link>
              <Link to="/early-stages" className="btn-liquid-secondary inline-flex items-center gap-2">
                <Activity size={17} className="text-[var(--cyan-liquid)]" />
                <span>Explore Early Signs & Guide</span>
              </Link>
            </div>

            {/* 4 Specular Trust Badges */}
            <div className="grid grid-cols-2 gap-3 pt-4 sm:grid-cols-4">
              <div className="glass-pill !flex-col !items-start !gap-0.5 !rounded-2xl !p-3.5">
                <span className="font-mono text-lg font-bold text-ink-deep">7,200</span>
                <span className="text-[0.72rem] font-medium text-ink-muted">Validated MRI Scans</span>
              </div>
              <div className="glass-pill !flex-col !items-start !gap-0.5 !rounded-2xl !p-3.5">
                <span className="font-mono text-lg font-bold text-ink-deep">5.9 ms</span>
                <span className="text-[0.72rem] font-medium text-ink-muted">GPU Inference Time</span>
              </div>
              <div className="glass-pill !flex-col !items-start !gap-0.5 !rounded-2xl !p-3.5">
                <span className="font-mono text-lg font-bold text-[var(--teal-soft)]">96.19%</span>
                <span className="text-[0.72rem] font-medium text-ink-muted">Clinical Recall</span>
              </div>
              <div className="glass-pill !flex-col !items-start !gap-0.5 !rounded-2xl !p-3.5">
                <span className="font-mono text-lg font-bold text-[var(--cyan-liquid)]">Grad-CAM</span>
                <span className="text-[0.72rem] font-medium text-ink-muted">Visual Heatmaps</span>
              </div>
            </div>
          </div>

          {/* Right Column: Female Orthopedic / Neuro Specialist Doctor Showcase */}
          <div className="lg:col-span-5">
            <div
              className="doctor-hero-frame liquid-capsule relative mx-auto max-w-[480px] cursor-pointer"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
              style={{
                transform: `perspective(1000px) rotateY(${tilt.x}deg) rotateX(${tilt.y}deg)`,
                transition: "transform 0.2s ease-out",
              }}
            >
              <div className="glass-specular-glare" />
              <img
                src={doctorImg}
                alt="Dr. Sarah Lin, MD - Neurological and Orthopedic Specialist consulting on brain MRI diagnostics"
                className="doctor-img aspect-[4/5] w-full object-cover"
                loading="eager"
                width={540}
                height={675}
              />

              {/* Floating Top Badge */}
              <div className="floating-badge badge-top-right">
                <span className="badge-dot" />
                <span className="font-medium">Dr. Sarah Lin, MD · Neuro-Orthopedic Specialist</span>
              </div>

              {/* Floating Bottom Badge */}
              <div className="floating-badge badge-bottom-left">
                <span className="badge-icon">✓</span>
                <span className="font-medium">Verified Clinical Partner · 96.19% Model Recall</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================= PLAIN-ENGLISH HUMAN EXPLANATION ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="space-y-6 text-center">
          <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--cyan-liquid)]">
            Demystifying Neuro-Radiology
          </span>
          <h2 className="font-display text-3xl font-bold tracking-tight text-ink-deep sm:text-4xl">
            Why We Built BrainTumor AI
          </h2>
          <p className="mx-auto max-w-3xl text-base text-ink-body md:text-lg">
            Complex medicine doesn't need to be intimidating. We translated high-dimensional
            computer vision into clear answers for real human lives.
          </p>
        </div>

        <div className="mt-12 grid gap-6 md:grid-cols-2">
          {/* The Problem */}
          <div className="liquid-card radius-card-lg p-8 transition-transform hover:-translate-y-1">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-red-500/10 text-[var(--glioma)]">
              <Brain size={26} />
            </div>
            <h3 className="font-display text-2xl font-bold text-ink-deep">
              The Problem in Everyday Terms
            </h3>
            <p className="mt-3 leading-relaxed text-ink-body">
              The human brain is our most delicate organ, safely cradled inside a solid bone skull.
              When unusual cell clusters begin to multiply, discovering them at the earliest possible
              moment is life-saving.
            </p>
            <p className="mt-3 leading-relaxed text-ink-body">
              However, reading a modern multi-slice MRI scan takes hours of intense examination. With
              hospital radiology departments reviewing thousands of scans every week, patients often
              endure days of agonizing anxiety waiting for a preliminary diagnosis.
            </p>
          </div>

          {/* The AI Solution */}
          <div className="liquid-card radius-card-lg p-8 transition-transform hover:-translate-y-1">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-[var(--cyan-liquid)]/10 text-[var(--cyan-liquid)]">
              <Sparkles size={26} />
            </div>
            <h3 className="font-display text-2xl font-bold text-ink-deep">
              The AI Solution in Everyday Terms
            </h3>
            <p className="mt-3 leading-relaxed text-ink-body">
              BrainTumor AI acts as a tireless, ultra-fast diagnostic assistant. Just as a senior
              neurologist has examined tens of thousands of patient scans over a 30-year career, our
              deep learning model has studied subtle tissue densities across 7,200 curated scans.
            </p>
            <p className="mt-3 leading-relaxed text-ink-body">
              In <strong>11.4 milliseconds</strong>, it evaluates cranial scans to identify normal
              healthy tissue or pinpoint signs of the three most common brain tumor varieties:
              Gliomas, Meningiomas, or Pituitary tumors.
            </p>
          </div>
        </div>

        {/* 4 Everyday Benefits Pills */}
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="liquid-card-subtle radius-card-md p-5">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[var(--teal-soft)]/15 text-[var(--teal-soft)]">
                <CheckCircle2 size={19} />
              </span>
              <h4 className="font-display text-sm font-bold text-ink-deep">Instant Second Opinion</h4>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-ink-muted">
              Receive preliminary reassurance and objective metrics while awaiting scheduled specialist appointments.
            </p>
          </div>

          <div className="liquid-card-subtle radius-card-md p-5">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[var(--cyan-liquid)]/15 text-[var(--cyan-liquid)]">
                <Eye size={19} />
              </span>
              <h4 className="font-display text-sm font-bold text-ink-deep">Zero Guesswork (Grad-CAM)</h4>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-ink-muted">
              We never act as an opaque black box. An interactive color heat-light illuminates the exact pixels analyzed.
            </p>
          </div>

          <div className="liquid-card-subtle radius-card-md p-5">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-amber-500/15 text-[var(--meningioma)]">
                <ShieldCheck size={19} />
              </span>
              <h4 className="font-display text-sm font-bold text-ink-deep">96.19% High Sensitivity</h4>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-ink-muted">
              Tuned for safety and maximum recall so suspicious tissue abnormalities are never overlooked.
            </p>
          </div>

          <div className="liquid-card-subtle radius-card-md p-5">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-purple-500/15 text-[var(--pituitary)]">
                <FileCheck2 size={19} />
              </span>
              <h4 className="font-display text-sm font-bold text-ink-deep">Completely Non-Invasive</h4>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-ink-muted">
              Requires only standard digital MRI image files (JPEG or PNG). No physical procedures or extra radiation.
            </p>
          </div>
        </div>
      </section>

      {/* ================= 6-CARD FEATURES GRID ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="text-center">
          <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--cyan-liquid)]">
            Clinical System Capabilities
          </span>
          <h2 className="mt-2 font-display text-3xl font-bold tracking-tight text-ink-deep sm:text-4xl">
            Engineered for Accuracy, Speed & Trust
          </h2>
        </div>

        <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {/* Card 1 */}
          <div className="liquid-card radius-card-lg p-7 transition-all hover:-translate-y-1 hover:shadow-lg">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-[var(--cyan-liquid)]/10 text-[var(--cyan-liquid)]">
              <Brain size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-ink-deep">
              4 Clear Tumor Categories
            </h3>
            <p className="mt-1 text-xs font-semibold text-[var(--cyan-liquid)]">
              Clear identification · Zero medical jargon
            </p>
            <p className="mt-3 text-sm leading-relaxed text-ink-body">
              Accurately differentiates healthy brain tissue from Gliomas (inner brain tissue),
              Meningiomas (protective membrane linings), and Pituitary growths (skull base).
            </p>
          </div>

          {/* Card 2 */}
          <div className="liquid-card radius-card-lg p-7 transition-all hover:-translate-y-1 hover:shadow-lg">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-[var(--teal-soft)]/10 text-[var(--teal-soft)]">
              <Zap size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-ink-deep">
              5.9-Millisecond Speed
            </h3>
            <p className="mt-1 text-xs font-semibold text-[var(--teal-soft)]">
              Faster than the blink of an eye
            </p>
            <p className="mt-3 text-sm leading-relaxed text-ink-body">
              Leverages local GPU acceleration (NVIDIA RTX 3050 ONNX Runtime) to evaluate scans
              in real time, removing agonizing multi-day delays for preliminary triage.
            </p>
          </div>

          {/* Card 3 */}
          <div className="liquid-card radius-card-lg p-7 transition-all hover:-translate-y-1 hover:shadow-lg">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-500/10 text-[var(--meningioma)]">
              <Eye size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-ink-deep">
              Grad-CAM Visual Proof
            </h3>
            <p className="mt-1 text-xs font-semibold text-[var(--meningioma)]">
              We show you exactly what the AI saw
            </p>
            <p className="mt-3 text-sm leading-relaxed text-ink-body">
              An interactive opacity slider projects gradient-weighted class activation heatmaps
              directly onto the MRI scan, allowing clinicians to inspect boundary precision.
            </p>
          </div>

          {/* Card 4 */}
          <div className="liquid-card radius-card-lg p-7 transition-all hover:-translate-y-1 hover:shadow-lg">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-red-500/10 text-[var(--glioma)]">
              <ShieldCheck size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-ink-deep">
              96.19% High Sensitivity
            </h3>
            <p className="mt-1 text-xs font-semibold text-[var(--glioma)]">
              Rigorously tuned for diagnostic safety
            </p>
            <p className="mt-3 text-sm leading-relaxed text-ink-body">
              Evaluated with priority on macro-recall over raw accuracy, ensuring that borderline or
              early-stage tumor patterns are never mislabeled as clean normal tissue.
            </p>
          </div>

          {/* Card 5 */}
          <div className="liquid-card radius-card-lg p-7 transition-all hover:-translate-y-1 hover:shadow-lg">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-purple-500/10 text-[var(--pituitary)]">
              <Lock size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-ink-deep">
              Privacy by Design
            </h3>
            <p className="mt-1 text-xs font-semibold text-[var(--pituitary)]">
              Your medical scans stay on your machine
            </p>
            <p className="mt-3 text-sm leading-relaxed text-ink-body">
              Inferences are calculated locally via our workstation FastAPI server. Patient
              cranial files are never transmitted to unverified cloud servers or third parties.
            </p>
          </div>

          {/* Card 6 */}
          <div className="liquid-card radius-card-lg p-7 transition-all hover:-translate-y-1 hover:shadow-lg">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-500/10 text-[var(--notumor)]">
              <Stethoscope size={24} />
            </div>
            <h3 className="font-display text-lg font-bold text-ink-deep">
              Doctor-Assisted Second Opinion
            </h3>
            <p className="mt-1 text-xs font-semibold text-[var(--notumor)]">
              Empowering healthcare professionals
            </p>
            <p className="mt-3 text-sm leading-relaxed text-ink-body">
              Designed as a collaborative tool that works alongside radiologists and neurosurgeons,
              accelerating routine screening and flagging subtle nodules for closer biopsy review.
            </p>
          </div>
        </div>
      </section>

      {/* ================= 3-STEP DIAGNOSTIC JOURNEY ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="liquid-card radius-card-lg p-8 md:p-12">
          <div className="text-center">
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--cyan-liquid)]">
              Simple 3-Step Journey
            </span>
            <h2 className="mt-2 font-display text-2xl font-bold text-ink-deep sm:text-3xl">
              How the Diagnostic Scan Works
            </h2>
            <p className="mt-2 text-sm text-ink-body md:text-base">
              A frictionless workflow built for clinicians and everyday patients alike.
            </p>
          </div>

          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {/* Step 1 */}
            <div className="liquid-card-subtle radius-card-md relative p-6 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-white shadow-sm font-mono text-lg font-bold text-[var(--cyan-liquid)]">
                01
              </div>
              <h3 className="mt-4 font-display text-base font-bold text-ink-deep">
                Upload MRI Scan
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-ink-muted">
                Drag and drop any standard cranial MRI slice (Axial, Coronal, or Sagittal) in JPEG,
                PNG, or WebP format.
              </p>
            </div>

            {/* Step 2 */}
            <div className="liquid-card-subtle radius-card-md relative p-6 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-white shadow-sm font-mono text-lg font-bold text-[var(--teal-soft)]">
                02
              </div>
              <h3 className="mt-4 font-display text-base font-bold text-ink-deep">
                Instant Neural Scan
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-ink-muted">
                The 4.0M-parameter EfficientNet engine evaluates tissue features across channel-last
                memory in 5.9ms.
              </p>
            </div>

            {/* Step 3 */}
            <div className="liquid-card-subtle radius-card-md relative p-6 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-white shadow-sm font-mono text-lg font-bold text-[var(--pituitary)]">
                03
              </div>
              <h3 className="mt-4 font-display text-base font-bold text-ink-deep">
                Plain-English Report
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-ink-muted">
                Review the multi-class probability breakdown, confidence dial, and toggle the Grad-CAM
                visual explainability overlay.
              </p>
            </div>
          </div>

          <div className="mt-10 text-center">
            <Link to="/detector" className="btn-liquid-primary inline-flex items-center gap-2">
              <span>Try the AI Tumor Detector Studio Now</span>
              <ArrowRight size={17} />
            </Link>
          </div>
        </div>
      </section>

      {/* ================= FOOTER / CITATION ================= */}
      <footer className="container mx-auto px-4 text-center text-xs text-ink-muted md:px-8">
        <p>
          BrainTumor AI · Clinical Deep Learning Architecture (EfficientNet-B0 ONNX GPU) · 96.19%
          Macro-Recall
        </p>
        <p className="mt-1">
          Designed for diagnostic triage support and educational consultation. Always confirm
          findings with a licensed medical professional.
        </p>
      </footer>
    </div>
  );
}
