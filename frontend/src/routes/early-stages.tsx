import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import {
  Heart,
  Eye,
  Brain,
  Shield,
  HelpCircle,
  Clock,
  Sparkles,
  CheckSquare,
  AlertCircle,
  FileText,
  Printer,
  ChevronRight,
} from "lucide-react";
import anatomyImg from "@/assets/brain_anatomy_symptom_map.jpg";
import progressionImg from "@/assets/early_vs_late_progression.jpg";
import mriSuiteImg from "@/assets/compassionate_mri_suite.jpg";
import symptomMatrixImg from "@/assets/symptom_matrix_illustrations.jpg";
import cellularImg from "@/assets/cellular_microenvironment.jpg";

export const Route = createFileRoute("/early-stages")({
  component: EarlyStagesPage,
});

/* Interactive Lobe Data */
const LOBES = [
  {
    name: "Frontal Lobe",
    role: "Speech, Motor Control & Executive Judgment",
    symptom: "Subtle personality shifts, delayed word finding, or slight arm weakness.",
    color: "var(--cyan-liquid)",
  },
  {
    name: "Temporal Lobe",
    role: "Memory Formation, Hearing & Language",
    symptom: "Short-term memory pauses, deja vu feelings, or difficulty understanding speech.",
    color: "var(--pituitary)",
  },
  {
    name: "Parietal Lobe",
    role: "Sensory Processing & Spatial Navigation",
    symptom: "Bumping into doorframes, tingling in fingers, or difficulty with writing.",
    color: "var(--teal-soft)",
  },
  {
    name: "Occipital Lobe",
    role: "Visual Perception & Color Interpretation",
    symptom: "Unexplained blurry spots, flashing visual auras, or peripheral blind areas.",
    color: "var(--meningioma)",
  },
  {
    name: "Cerebellum & Brainstem",
    role: "Balance, Coordination & Involuntary Vital Reflexes",
    symptom: "Unsteadiness while walking, dizziness, or morning nausea when standing upright.",
    color: "var(--glioma)",
  },
  {
    name: "Pituitary Gland (Skull Base)",
    role: "Master Hormonal Balance & Optic Nerve Chiasm",
    symptom: "Peripheral tunnel vision, unexplained chronic fatigue, or extreme thirst shifts.",
    color: "#6366f1",
  },
];

export function EarlyStagesPage() {
  const [selectedLobe, setSelectedLobe] = useState(0);
  const [checkedQuestions, setCheckedQuestions] = useState<Record<number, boolean>>({});

  const toggleQuestion = (idx: number) => {
    setCheckedQuestions((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-20 py-6 md:py-10">
      {/* ================= PAGE HEADER ================= */}
      <section className="container mx-auto px-4 text-center md:px-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/60 px-3.5 py-1.5 shadow-sm backdrop-blur-md">
          <span className="h-2 w-2 rounded-full bg-[var(--teal-soft)] animate-pulse" />
          <span className="font-mono text-xs font-semibold text-ink-deep">
            Patient Guide & Clinical Education Suite
          </span>
        </div>
        <h1 className="mt-4 font-display text-3xl font-extrabold tracking-tight text-ink-deep sm:text-4xl lg:text-5xl">
          Early Stages & Patient Understanding Guide
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-base text-ink-body md:text-lg">
          A gentle, reassuring exploration of brain health, subtle early warning signals, and why
          catching abnormalities early preserves full quality of life.
        </p>
      </section>

      {/* ================= GENTLE REASSURING METAPHOR ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="liquid-card radius-card-lg relative overflow-hidden p-8 md:p-12">
          <div className="max-w-3xl space-y-4">
            <div className="inline-flex items-center gap-2 rounded-xl bg-[var(--teal-soft)]/10 px-3 py-1 text-xs font-bold text-[var(--teal-soft)]">
              <Heart size={16} />
              <span>A Gentle Reassurance</span>
            </div>
            <h2 className="font-display text-2xl font-bold text-ink-deep sm:text-3xl">
              Understanding Your Brain as a Safe, Protected Space
            </h2>
            <p className="text-base leading-relaxed text-ink-body md:text-lg">
              Think of your skull as a cozy, safe home built out of solid bone. Inside, your brain
              fits comfortably, surrounded by gentle protective fluid cushions called cerebrospinal
              fluid.
            </p>
            <p className="text-base leading-relaxed text-ink-body">
              Because the skull bone cannot stretch or expand, even a tiny new visitor—a cluster of
              cells multiplying where it shouldn't—begins to take up space. By detecting this
              cluster when it is no larger than a grain of rice, doctors can safely resolve it long
              before it creates pressure on delicate nerves.
            </p>
          </div>
        </div>
      </section>

      {/* ================= THE 4 TUMOR STAGES / WHO GRADES EXPLAINED ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="text-center">
          <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--cyan-liquid)]">
            World Health Organization (WHO) Grading System
          </span>
          <h2 className="mt-2 font-display text-2xl font-bold text-ink-deep sm:text-3xl">
            The 4 Tumor Stages Explained in Plain English
          </h2>
          <p className="mx-auto mt-2 max-w-2xl text-xs text-ink-muted sm:text-sm">
            Doctors categorize brain tumors by their cellular growth pace rather than traditional
            body staging.
          </p>
        </div>

        <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {/* Grade 1 */}
          <div className="liquid-card radius-card-md p-6">
            <span className="font-mono text-xs font-bold uppercase text-[var(--notumor)]">
              Grade I
            </span>
            <h3 className="mt-1 font-display text-xl font-bold text-ink-deep">
              The Slow Nodule
            </h3>
            <ul className="mt-4 space-y-2 text-xs leading-relaxed text-ink-body">
              <li>• Cells divide very slowly and remain localized.</li>
              <li>• Distinct, clean borders from surrounding brain tissue.</li>
              <li>• Frequently 100% curable through straightforward surgical removal.</li>
            </ul>
          </div>

          {/* Grade 2 */}
          <div className="liquid-card radius-card-md p-6">
            <span className="font-mono text-xs font-bold uppercase text-[var(--meningioma)]">
              Grade II
            </span>
            <h3 className="mt-1 font-display text-xl font-bold text-ink-deep">
              The Gentle Watch
            </h3>
            <ul className="mt-4 space-y-2 text-xs leading-relaxed text-ink-body">
              <li>• Cells look slightly unusual under a microscope.</li>
              <li>• May slowly extend into nearby normal tissue.</li>
              <li>• Managed with periodic annual MRI surveillance and targeted removal.</li>
            </ul>
          </div>

          {/* Grade 3 */}
          <div className="liquid-card radius-card-md p-6">
            <span className="font-mono text-xs font-bold uppercase text-[var(--pituitary)]">
              Grade III
            </span>
            <h3 className="mt-1 font-display text-xl font-bold text-ink-deep">
              The Active Crew
            </h3>
            <ul className="mt-4 space-y-2 text-xs leading-relaxed text-ink-body">
              <li>• Actively dividing cells requiring proactive treatment.</li>
              <li>• Requires combination of precision surgery and targeted radiation.</li>
              <li>• Responds strongly when identified before deep tissue infiltration.</li>
            </ul>
          </div>

          {/* Grade 4 */}
          <div className="liquid-card radius-card-md p-6">
            <span className="font-mono text-xs font-bold uppercase text-[var(--glioma)]">
              Grade IV
            </span>
            <h3 className="mt-1 font-display text-xl font-bold text-ink-deep">
              The Rapid Spurt
            </h3>
            <ul className="mt-4 space-y-2 text-xs leading-relaxed text-ink-body">
              <li>• Rapidly growing cells forming their own blood vessels (angiogenesis).</li>
              <li>• Requires immediate, multi-disciplinary neuro-oncological care.</li>
              <li>• Contemporary clinical trials offer targeted molecular therapies.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* ================= ARTWORK 1: INTERACTIVE BRAIN ANATOMY & SYMPTOM MAP ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="liquid-card radius-card-lg p-6 md:p-10">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/60 pb-4">
            <div>
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--cyan-liquid)]">
                Interactive Cranial Map
              </span>
              <h2 className="font-display text-2xl font-bold text-ink-deep sm:text-3xl">
                Brain Anatomy & Symptom Correlation
              </h2>
            </div>
            <span className="text-xs text-ink-muted">
              Select any lobe below to reveal its functional domain
            </span>
          </div>

          <div className="mt-8 grid gap-8 lg:grid-cols-12 lg:items-center">
            {/* 3D Brain Visual (7 cols) */}
            <div className="relative overflow-hidden rounded-2xl border border-white/80 bg-white/40 shadow-inner lg:col-span-7">
              <img
                src={anatomyImg}
                alt="Translucent 3D Brain Anatomy with mapped lobes"
                className="w-full object-contain"
                loading="lazy"
                width={700}
                height={525}
              />
              <div className="absolute bottom-3 left-3 rounded-lg bg-white/80 px-3 py-1 font-mono text-[0.7rem] text-ink-deep backdrop-blur-md">
                Active View: <strong>{LOBES[selectedLobe]?.name}</strong>
              </div>
            </div>

            {/* Interactive Lobe Selector (5 cols) */}
            <div className="space-y-3 lg:col-span-5">
              {LOBES.map((lobe, idx) => {
                const isSelected = selectedLobe === idx;
                return (
                  <div
                    key={lobe.name}
                    onClick={() => setSelectedLobe(idx)}
                    className={`cursor-pointer rounded-2xl border p-4 transition-all ${isSelected
                        ? "border-white bg-white/80 shadow-md translate-x-1"
                        : "border-transparent bg-white/40 hover:bg-white/60"
                      }`}
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className="font-display text-sm font-bold"
                        style={{ color: lobe.color }}
                      >
                        {lobe.name}
                      </span>
                      <ChevronRight
                        size={16}
                        className={`transition-transform ${isSelected ? "rotate-90 text-ink-deep" : "text-ink-muted"
                          }`}
                      />
                    </div>
                    <p className="mt-1 text-xs font-medium text-ink-deep">{lobe.role}</p>
                    {isSelected && (
                      <p className="mt-2 text-xs leading-relaxed text-ink-muted border-t border-white/60 pt-2">
                        <strong>Early Signal:</strong> {lobe.symptom}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* ================= ARTWORK 2: THE GOLDEN WINDOW & PROGRESSION COMPARISON ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="grid gap-8 lg:grid-cols-12 lg:items-center">
          {/* Text (5 cols) */}
          <div className="space-y-4 lg:col-span-5">
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--teal-soft)]">
              Early Detection Saves Lives
            </span>
            <h2 className="font-display text-3xl font-bold tracking-tight text-ink-deep">
              The "Golden Window" of Early Detection
            </h2>
            <p className="text-sm leading-relaxed text-ink-body">
              When caught in its earliest stage (under 1.5 cm), a brain tumor has clear, distinct
              tissue boundaries. This allows neurosurgeons to perform safe, minimally-invasive
              resections while fully preserving speech, memory, and motor functions.
            </p>
            <div className="space-y-2 pt-2 text-xs text-ink-body">
              <div className="flex items-start gap-2">
                <CheckSquare size={16} className="mt-0.5 text-[var(--teal-soft)] shrink-0" />
                <span>
                  <strong>Surgical Margin Safety:</strong> Minimal contact with critical cranial
                  arteries and veins.
                </span>
              </div>
              <div className="flex items-start gap-2">
                <CheckSquare size={16} className="mt-0.5 text-[var(--teal-soft)] shrink-0" />
                <span>
                  <strong>Targeted Radiosurgery:</strong> High-precision Gamma Knife therapy can
                  sterilize small nodules in a single outpatient session.
                </span>
              </div>
              <div className="flex items-start gap-2">
                <CheckSquare size={16} className="mt-0.5 text-[var(--teal-soft)] shrink-0" />
                <span>
                  <strong>Zero Mass Effect:</strong> Prevents brain midline shift and fluid pressure
                  buildup.
                </span>
              </div>
            </div>
          </div>

          {/* Progression Image (7 cols) */}
          <div className="liquid-card radius-card-lg overflow-hidden p-4 lg:col-span-7">
            <img
              src={progressionImg}
              alt="Side-by-side comparison of Early Stage 1 nodule vs Advanced Stage 4 tumor"
              className="w-full rounded-xl object-cover"
              loading="lazy"
              width={720}
              height={405}
            />
            <div className="mt-3 flex items-center justify-between px-2 text-xs text-ink-muted">
              <span>Stage 1: Localized &lt;1cm nodule</span>
              <span className="font-semibold text-[var(--glioma)]">
                Stage 4: Midline shift & edema
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ================= ARTWORK 3: 4-PANEL SYMPTOM MATRIX ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="text-center">
          <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--cyan-liquid)]">
            Recognizing Subtle Signals
          </span>
          <h2 className="mt-2 font-display text-2xl font-bold text-ink-deep sm:text-3xl">
            Everyday Early Warning Signs
          </h2>
          <p className="mx-auto mt-2 max-w-2xl text-xs text-ink-muted sm:text-sm">
            Most headaches are simply tension or dehydration. However, these specific subtle patterns
            warrant a gentle conversation with your doctor.
          </p>
        </div>

        <div className="mt-8 liquid-card radius-card-lg p-6">
          <div className="grid gap-6 lg:grid-cols-12 lg:items-center">
            {/* 4-Panel Illustration */}
            <div className="overflow-hidden rounded-2xl border border-white/80 bg-white/40 lg:col-span-6">
              <img
                src={symptomMatrixImg}
                alt="4-Panel illustration of early warning symptoms"
                className="w-full object-cover"
                loading="lazy"
                width={600}
                height={340}
              />
            </div>

            {/* Symptom Cards Grid */}
            <div className="grid gap-3 sm:grid-cols-2 lg:col-span-6">
              <div className="liquid-card-subtle radius-card-md p-4">
                <h3 className="font-display text-xs font-bold text-ink-deep">
                  1. The Morning Headache Clue
                </h3>
                <p className="mt-1 text-[0.78rem] leading-relaxed text-ink-muted">
                  Noticeably stronger upon waking up or when coughing/bending, easing after standing
                  upright for an hour.
                </p>
              </div>

              <div className="liquid-card-subtle radius-card-md p-4">
                <h3 className="font-display text-xs font-bold text-ink-deep">
                  2. Peripheral Vision "Glitch"
                </h3>
                <p className="mt-1 text-[0.78rem] leading-relaxed text-ink-muted">
                  Bumping into doorframes on one side, or noticing subtle blurriness in the outer
                  corners of your reading vision.
                </p>
              </div>

              <div className="liquid-card-subtle radius-card-md p-4">
                <h3 className="font-display text-xs font-bold text-ink-deep">
                  3. Fingertip Clumsiness & Twitches
                </h3>
                <p className="mt-1 text-[0.78rem] leading-relaxed text-ink-muted">
                  Unexpectedly dropping coffee mugs, slight stumbling on flat floors, or tiny involuntary
                  muscle twitches.
                </p>
              </div>

              <div className="liquid-card-subtle radius-card-md p-4">
                <h3 className="font-display text-xs font-bold text-ink-deep">
                  4. Word-Finding Memory Pauses
                </h3>
                <p className="mt-1 text-[0.78rem] leading-relaxed text-ink-muted">
                  Frequent "tip-of-the-tongue" pauses for simple everyday words, or sudden subtle
                  personality shifts noted by family.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================= ARTWORK 4 & 5: CELLULAR ENVIRONMENT & COMPASSIONATE MRI SUITE ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Cellular Environment */}
          <div className="liquid-card radius-card-lg p-6">
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--pituitary)]">
              Microscopic Clarity
            </span>
            <h3 className="mt-1 font-display text-xl font-bold text-ink-deep">
              How Tumors Begin at the Cellular Level
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-ink-muted">
              Healthy brain tissue consists of delicate star-shaped astrocytes and neural synapses.
              Computer vision learns to distinguish these natural patterns from early abnormal
              clusters.
            </p>
            <div className="mt-4 overflow-hidden rounded-2xl border border-white/80 bg-white/40">
              <img
                src={cellularImg}
                alt="Microscopic view of astrocytes and early abnormal dividing cells"
                className="aspect-video w-full object-cover"
                loading="lazy"
                width={600}
                height={340}
              />
            </div>
          </div>

          {/* Compassionate MRI Suite */}
          <div className="liquid-card radius-card-lg p-6">
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-[var(--teal-soft)]">
              Alleviating Anxiety
            </span>
            <h3 className="mt-1 font-display text-xl font-bold text-ink-deep">
              What to Expect During Your MRI
            </h3>
            <p className="mt-2 text-xs leading-relaxed text-ink-muted">
              Modern MRI suites are calm, sunlit, and spacious. You rest comfortably with headphones
              listening to music while a friendly technician guides you through every step.
            </p>
            <div className="mt-4 overflow-hidden rounded-2xl border border-white/80 bg-white/40">
              <img
                src={mriSuiteImg}
                alt="Compassionate patient lying comfortably in modern open MRI with friendly doctor"
                className="aspect-video w-full object-cover"
                loading="lazy"
                width={600}
                height={340}
              />
            </div>
          </div>
        </div>
      </section>

      {/* ================= PATIENT ACTION ROADMAP & DOCTOR CHECKLIST ================= */}
      <section className="container mx-auto px-4 md:px-8">
        <div className="grid gap-8 lg:grid-cols-12">
          {/* Roadmap (6 cols) */}
          <div className="liquid-card radius-card-lg p-6 lg:col-span-6">
            <div className="flex items-center gap-2 border-b border-white/60 pb-3">
              <AlertCircle size={19} className="text-[var(--cyan-liquid)]" />
              <h3 className="font-display text-base font-bold text-ink-deep">
                Patient Action Roadmap
              </h3>
            </div>
            <div className="mt-4 space-y-4">
              <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
                <span className="font-mono text-[0.7rem] font-bold uppercase text-[var(--notumor)]">
                  Green Tier · Routine Care
                </span>
                <p className="mt-1 text-xs text-ink-body">
                  Occasional tension headaches with stress or fatigue. Schedule a standard annual
                  checkup with your family physician.
                </p>
              </div>

              <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
                <span className="font-mono text-[0.7rem] font-bold uppercase text-[var(--meningioma)]">
                  Yellow Tier · Medical Consultation
                </span>
                <p className="mt-1 text-xs text-ink-body">
                  Morning-specific headaches accompanied by slight visual blur or coordination pauses.
                  Request a non-urgent cranial MRI with contrast within 2 weeks.
                </p>
              </div>

              <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4">
                <span className="font-mono text-[0.7rem] font-bold uppercase text-[var(--glioma)]">
                  Red Tier · Immediate Triage
                </span>
                <p className="mt-1 text-xs text-ink-body">
                  Sudden onset seizure, sudden loss of arm sensation, or severe walking imbalance.
                  Proceed immediately to the emergency department for neuro-imaging.
                </p>
              </div>
            </div>
          </div>

          {/* Doctor Consultation Checklist (6 cols) */}
          <div className="liquid-card radius-card-lg p-6 lg:col-span-6">
            <div className="flex items-center justify-between border-b border-white/60 pb-3">
              <div className="flex items-center gap-2">
                <FileText size={19} className="text-[var(--teal-soft)]" />
                <h3 className="font-display text-base font-bold text-ink-deep">
                  Doctor Consultation Pocket Checklist
                </h3>
              </div>
              <button
                onClick={handlePrint}
                className="inline-flex items-center gap-1 rounded-lg bg-white/70 px-2.5 py-1 text-xs font-semibold text-ink-deep hover:bg-white shadow-sm"
              >
                <Printer size={13} />
                <span>Print</span>
              </button>
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              Five essential questions to ask your physician when reviewing a cranial MRI scan:
            </p>
            <div className="mt-4 space-y-2.5">
              {[
                "1. Does the scan show any localized mass effect, tissue swelling, or midline shift?",
                "2. Are the margins of the tissue sharp and well-defined, or diffuse?",
                "3. Is contrast enhancement (Gadolinium) recommended for a clearer boundary?",
                "4. What specific brain lobe is adjacent to the region, and does it affect my motor or speech functions?",
                "5. What is the recommended follow-up schedule (e.g. 6-month repeat scan vs neurosurgical consultation)?",
              ].map((q, idx) => (
                <div
                  key={idx}
                  onClick={() => toggleQuestion(idx)}
                  className={`flex cursor-pointer items-start gap-2.5 rounded-xl border p-3 text-xs transition ${checkedQuestions[idx]
                      ? "border-[var(--teal-soft)]/50 bg-[var(--teal-soft)]/10 text-ink-deep"
                      : "border-white/60 bg-white/40 text-ink-body hover:bg-white/70"
                    }`}
                >
                  <input
                    type="checkbox"
                    checked={!!checkedQuestions[idx]}
                    onChange={() => { }}
                    className="mt-0.5 accent-[var(--teal-soft)] cursor-pointer"
                  />
                  <span>{q}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
