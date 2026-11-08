import { AlertTriangle, BookOpen, Radio, ServerCrash, Wifi } from "lucide-react";
import type { LogEntry, LogKind } from "@/lib/quaviz/types";
import type { LucideIcon } from "lucide-react";

const kindClass: Record<LogKind, string> = {
  alert: "text-alarm",
  explain: "text-primary",
  telemetry: "text-muted-foreground",
  system: "text-ok",
  error: "text-alarm",
};

const kindIcon: Record<LogKind, LucideIcon> = {
  alert: AlertTriangle,
  explain: BookOpen,
  telemetry: Radio,
  system: Wifi,
  error: ServerCrash,
};

function clock(at: number) {
  const d = new Date(at);
  return d.toISOString().slice(11, 19);
}

export function LiveLog({
  entries,
  onPickAsset,
}: {
  entries: LogEntry[];
  onPickAsset?: (assetId: string) => void;
}) {
  return (
    <aside className="panel flex h-full min-h-0 flex-col overflow-hidden" aria-label="Live event log">
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-ok opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-ok" />
          </span>
          <h2 className="label-caps !mb-0">Live log</h2>
        </div>
        <span className="font-mono text-[0.65rem] text-muted-foreground">{entries.length}</span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto px-2 py-1 font-mono text-xs">
        {entries.length === 0 ? (
          <p className="p-4 text-center text-muted-foreground">Waiting for events…</p>
        ) : (
          <ul className="space-y-0.5">
            {entries.map((e) => {
              const Icon = kindIcon[e.kind];
              return (
                <li
                  key={e.id}
                  className={`animate-row-in rounded-sm px-2 py-1.5 hover:bg-secondary/60 ${e.asset_id && onPickAsset ? "cursor-pointer" : ""}`}
                  onClick={() => e.asset_id && onPickAsset?.(e.asset_id)}
                  onKeyDown={(ev) => {
                    if ((ev.key === "Enter" || ev.key === " ") && e.asset_id && onPickAsset) {
                      ev.preventDefault();
                      onPickAsset(e.asset_id);
                    }
                  }}
                  tabIndex={e.asset_id ? 0 : undefined}
                >
                  <div className="flex items-start gap-2">
                    <span className="shrink-0 pt-0.5 text-muted-foreground">{clock(e.at)}</span>
                    <Icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${kindClass[e.kind]}`} aria-hidden />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-foreground/90">{e.title}</div>
                      {e.detail && (
                        <div className="mt-0.5 truncate text-muted-foreground">{e.detail}</div>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </aside>
  );
}
