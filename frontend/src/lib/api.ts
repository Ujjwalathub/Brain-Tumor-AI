/**
 * REST client for the local FastAPI inference engine (/predict, /health).
 * When the engine is unreachable, the Detector Studio falls back to a
 * clearly-labeled simulated demo mode so the UI remains fully explorable.
 */

export const API_BASE: string =
  (import.meta.env["VITE_API_BASE"] as string | undefined) ?? "http://localhost:8000";

export type Health = {
  status: string;
  model_loaded: boolean;
  device: string;
  model_name: string;
};

export type Prediction = {
  class_name: string;
  confidence: number;
  probabilities: Record<string, number>;
  inference_time_ms: number;
};

/**
 * Error thrown when the backend received the request but rejected it
 * (e.g. color image on a grayscale-only pipeline, corrupt file).
 * Distinct from a network failure so the UI can surface it instead of
 * silently falling back to simulations.
 */
export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function checkHealth(timeoutMs = 3000): Promise<Health> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
    return (await res.json()) as Health;
  } finally {
    clearTimeout(timer);
  }
}

export async function runTumorInference(imageFile: File): Promise<Prediction> {
  const formData = new FormData();
  formData.append("file", imageFile);

  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });

  if (!res.ok) {
    let detail = "Diagnostic scan failed";
    try {
      const err = await res.json();
      // The API's exception handler returns the message in `error`,
      // with `detail` as an empty companion field.
      detail = err.error ?? err.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }

  return (await res.json()) as Prediction;
}

/* ---------- Simulated demo mode (engine offline) ---------- */

function hashString(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Deterministic, plausible-looking result derived from the file name so the
 * demo is reproducible while still showing all four classes across uploads.
 */
export function simulatePrediction(file: File): Prediction {
  const h = hashString(file.name + file.size);
  const pick = h % 100;
  const className =
    pick < 40 ? "Glioma" : pick < 63 ? "Meningioma" : pick < 82 ? "Pituitary" : "No Tumor";

  const confidence = 0.92 + ((h >>> 8) % 700) / 10000; // 0.92 - 0.9899
  const remaining = 1 - confidence;
  const others = ["Glioma", "Meningioma", "No Tumor", "Pituitary"].filter((c) => c !== className);
  const weights = [((h >>> 12) % 45) + 30, ((h >>> 16) % 35) + 15, ((h >>> 20) % 25) + 5];
  const wSum = weights.reduce((a, b) => a + b, 0);

  const probabilities: Record<string, number> = { [className]: confidence };
  others.forEach((c, i) => {
    const w = weights[i] ?? 10;
    probabilities[c] = Number(((remaining * w) / wSum).toFixed(4));
  });

  return {
    class_name: className,
    confidence: Number(confidence.toFixed(4)),
    probabilities,
    inference_time_ms: Number((10.2 + ((h >>> 4) % 40) / 10).toFixed(1)), // 10.2 - 14.1 ms
  };
}

/** Seeded Grad-CAM focus point (percentages) for the demo heatmap overlay. */
export function gradcamFocus(file: File): { x: number; y: number } {
  const h = hashString(file.name);
  return { x: 28 + (h % 44), y: 26 + ((h >>> 9) % 40) };
}
