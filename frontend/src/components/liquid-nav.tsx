import { Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { API_BASE, checkHealth } from "@/lib/api";
import { BrainCircuit } from "lucide-react";

const NAV_LINKS = [
  { to: "/" as const, label: "Overview" },
  { to: "/detector" as const, label: "Detector" },
  { to: "/performance" as const, label: "Performance" },
  { to: "/early-stages" as const, label: "Early Stages" },
];

export function LiquidNav() {
  const [status, setStatus] = useState<"checking" | "online" | "offline">("checking");
  const [device, setDevice] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    checkHealth()
      .then((h) => {
        if (cancelled) return;
        setStatus("online");
        setDevice(h.device ?? "");
      })
      .catch(() => {
        if (!cancelled) setStatus("offline");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const dotColor =
    status === "online"
      ? "var(--notumor)"
      : status === "offline"
        ? "var(--meningioma)"
        : "var(--cyan-liquid)";
  const statusLabel =
    status === "online"
      ? `FastAPI Online${device ? ` · ${device}` : ""}`
      : status === "offline"
        ? "Demo Mode · Engine Offline"
        : "Connecting…";

  return (
    <nav className="nav-glass flex items-center justify-between gap-2 px-4 py-1.5" aria-label="Primary">
      <Link to="/" className="flex items-center gap-2 pr-2">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-[#0a1120] to-[#16243e] text-white shadow-sm">
          <BrainCircuit size={16} strokeWidth={2.2} />
        </span>
        <span className="font-display text-[0.92rem] font-extrabold tracking-tight text-ink-deep">
          BrainTumor AI
        </span>
      </Link>

      <div className="hidden items-center gap-1 md:flex" role="group">
        {NAV_LINKS.map((l, i) => (
          <Link
            key={l.to}
            to={l.to}
            className="nav-link"
            activeProps={{ className: "nav-link-active" }}
            activeOptions={{ exact: i === 0 }}
          >
            {l.label}
          </Link>
        ))}
      </div>

      <div className="flex items-center gap-2 md:hidden">
        {NAV_LINKS.slice(0, 2).map((l, i) => (
          <Link
            key={l.to}
            to={l.to}
            className="nav-link !px-3 text-[0.8rem]"
            activeProps={{ className: "nav-link-active" }}
            activeOptions={{ exact: i === 0 }}
          >
            {l.label}
          </Link>
        ))}
      </div>

      <div className="glass-pill hidden !py-1.5 lg:inline-flex">
        <span className="pulse-dot" style={{ background: dotColor }} aria-hidden />
        <span>{statusLabel}</span>
      </div>
    </nav>
  );
}
