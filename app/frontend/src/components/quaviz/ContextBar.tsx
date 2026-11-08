import {
  Activity,
  Droplets,
  Gauge,
  Ship,
  Thermometer,
  Waves,
} from "lucide-react";
import type { ConsoleState } from "@/lib/quaviz/store";
import type { ReactNode } from "react";

function Ctx({
  label,
  value,
  unit,
  icon,
}: {
  label: string;
  value: string;
  unit?: string;
  icon: ReactNode;
}) {
  return (
    <div className="flex min-w-[7rem] items-start gap-2.5">
      <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border/80 bg-secondary/60 text-primary">
        {icon}
      </span>
      <div>
        <div className="label-caps">{label}</div>
        <div className="font-mono text-lg font-semibold leading-tight tabular-nums">
          {value}
          {unit && <span className="ml-1 text-xs font-normal text-muted-foreground">{unit}</span>}
        </div>
      </div>
    </div>
  );
}

function fmt(n: number | null | undefined, digits = 1): string {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function ContextBar({
  s,
  counts,
}: {
  s: ConsoleState;
  counts: { red: number; amber: number; green: number; grey: number };
}) {
  const f = s.context?.fleet;
  const icon = "h-4 w-4";
  return (
    <header className="border-b border-border bg-card/80 px-5 py-3 backdrop-blur">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/15 text-primary">
            <Activity className="h-5 w-5" aria-hidden />
          </span>
          <div>
            <h1 className="text-xl font-bold tracking-wide leading-none">QuaViz</h1>
            <span className="label-caps">Fleet operations</span>
          </div>
          {s.mode === "demo" && (
            <span className="rounded-sm bg-warn-soft px-2 py-0.5 text-[0.65rem] font-bold tracking-wider text-warn">
              DEMO · SYNTHETIC
            </span>
          )}
        </div>
        <div className="flex flex-wrap gap-4 font-mono text-xs">
          <span className="inline-flex items-center gap-1.5 text-alarm">
            <span className="h-2 w-2 rounded-full bg-alarm" /> {counts.red} fault
          </span>
          <span className="inline-flex items-center gap-1.5 text-warn">
            <span className="h-2 w-2 rounded-full bg-warn" /> {counts.amber} watch
          </span>
          <span className="inline-flex items-center gap-1.5 text-ok">
            <span className="h-2 w-2 animate-pulse rounded-full bg-ok" /> {counts.green} healthy
          </span>
          <span className="inline-flex items-center gap-1.5 text-muted-foreground">
            <span className="h-2 w-2 rounded-full bg-idle" /> {counts.grey} idle
          </span>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-8 gap-y-3 border-t border-border/60 pt-3">
        <Ctx label="Schedule" value={f?.schedule_state ? String(f.schedule_state) : "—"} icon={<Ship className={icon} />} />
        <Ctx label="Duty demand" value={fmt(f?.duty_demand_pct)} unit="%" icon={<Gauge className={icon} />} />
        <Ctx label="Sea temp" value={fmt(f?.sea_temp_c)} unit="°C" icon={<Waves className={icon} />} />
        <Ctx label="Supply water" value={fmt(f?.supply_water_temp_c)} unit="°C" icon={<Droplets className={icon} />} />
        <Ctx label="Ambient" value={fmt(f?.ambient_temp_c)} unit="°C" icon={<Thermometer className={icon} />} />
        <Ctx label="Sea state Hs" value={fmt(f?.sea_state_hs_m)} unit="m" icon={<Waves className={icon} strokeWidth={1.5} />} />
        <Ctx label="Context assets" value={f?.n_assets != null ? String(f.n_assets) : "—"} icon={<Activity className={icon} />} />
      </div>
    </header>
  );
}
