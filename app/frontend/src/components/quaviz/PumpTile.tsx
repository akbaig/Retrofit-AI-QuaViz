import { humanFault, pct, relTime } from "@/lib/quaviz/status";
import type { Alert, AssetStatus, TileStatus } from "@/lib/quaviz/types";
import { PumpGlyph } from "./PumpGlyph";
import { tone } from "./tone";

interface Props {
  id: string;
  status: TileStatus;
  asset?: AssetStatus | undefined;
  latest?: Alert | undefined;
  selected: boolean;
  pulseKey?: number | undefined;
  now: number;
  onSelect: () => void;
}

export function PumpTile({ id, status, asset, latest, selected, pulseKey, now, onSelect }: Props) {
  const t = tone[status];
  const showFault = (status === "red" || status === "amber") && latest;
  const caption = showFault
    ? humanFault(latest.fault_type)
    : status === "green"
      ? "Running OK"
      : status === "grey"
        ? "Not seen"
        : asset?.messages
          ? "Listening"
          : "Unreachable";
  const when = latest ? relTime(latest.alert_time_utc, now) : relTime(asset?.last_timestamp_utc, now);
  const msgs = asset?.messages ?? 0;

  return (
    <button
      type="button"
      key={pulseKey}
      onClick={onSelect}
      aria-pressed={selected}
      aria-label={`${id}: ${t.label}, ${caption}`}
      className={`group relative flex flex-col overflow-hidden rounded-md border text-left transition-all duration-500 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-black/25 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring ${t.soft} ${t.border} ${selected ? "ring-2 ring-primary" : ""} ${pulseKey && now - pulseKey < 3000 ? "animate-alarm-pulse" : ""}`}
    >
      <span className={`h-1.5 w-full transition-colors duration-500 ${t.bar}`} />
      <span className="flex flex-1 gap-3 p-3">
        <span className="relative flex items-center justify-center">
          <span className={`absolute inset-0 rounded-full opacity-40 blur-md ${t.bar}`} />
          <PumpGlyph status={status} className="relative" />
        </span>
        <span className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="flex items-center justify-between gap-2">
            <span className="font-mono text-sm font-semibold tracking-tight">{id}</span>
            <span className={`text-[0.65rem] font-bold tracking-widest ${t.text}`}>{t.label}</span>
          </span>
          <span className={`truncate text-base font-semibold leading-tight ${showFault ? t.text : "text-foreground/85"}`}>
            {caption}
          </span>
          <span className="mt-auto flex items-end justify-between gap-2 font-mono text-xs text-muted-foreground">
            <span className="truncate">{when}</span>
            {showFault ? (
              <span className={`text-sm font-semibold ${t.text}`}>{pct(latest.confidence)}</span>
            ) : (
              <span className="tabular-nums opacity-70">{msgs ? `${msgs.toLocaleString()} msg` : "—"}</span>
            )}
          </span>
        </span>
      </span>
    </button>
  );
}
