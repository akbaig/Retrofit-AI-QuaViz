import { humanFault, pct, RED_CONFIDENCE } from "@/lib/quaviz/status";
import type { Alert } from "@/lib/quaviz/types";

interface Props {
  alerts: Alert[];
  filter: string | null;
  fresh: Record<string, true>;
  onPick: (a: Alert) => void;
  onClearFilter: () => void;
}

const fmt = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(5, 19).replace("T", " ");
};

export function AlertStream({ alerts, filter, fresh, onPick, onClearFilter }: Props) {
  const rows = filter ? alerts.filter((a) => a.asset_id === filter) : alerts;
  return (
    <section className="panel flex min-h-0 flex-col" aria-label="Alert stream">
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <h2 className="label-caps">Alert stream {filter && <>· {filter}</>}</h2>
        {filter && <button onClick={onClearFilter} className="text-xs text-primary hover:underline">Show all pumps</button>}
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {rows.length === 0 ? (
          <p className="p-6 text-center text-sm text-muted-foreground">No alerts yet — listening to telemetry.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card text-left">
              <tr className="label-caps">
                <th className="px-4 py-1.5 font-semibold">Time (UTC)</th><th className="px-2 font-semibold">Asset</th>
                <th className="px-2 font-semibold">Fault</th><th className="px-2 font-semibold">Class</th>
                <th className="px-2 text-right font-semibold">Conf.</th><th className="px-4 font-semibold">Action</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 200).map((a) => {
                const red = a.confidence >= RED_CONFIDENCE;
                return (
                  <tr key={a.id} tabIndex={0} onClick={() => onPick(a)}
                    onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onPick(a))}
                    className={`cursor-pointer border-t border-border/60 hover:bg-secondary focus-visible:bg-secondary focus-visible:outline-none ${fresh[a.id] ? "animate-row-in" : ""}`}>
                    <td className="whitespace-nowrap px-4 py-1.5 font-mono text-xs text-muted-foreground">{fmt(a.alert_time_utc)}</td>
                    <td className="whitespace-nowrap px-2 font-mono text-xs">{a.asset_id}</td>
                    <td className={`whitespace-nowrap px-2 font-semibold ${red ? "text-alarm" : "text-warn"}`}>{humanFault(a.fault_type)}</td>
                    <td className="px-2 text-xs text-muted-foreground">{a.event_class}</td>
                    <td className={`px-2 text-right font-mono ${red ? "text-alarm" : "text-warn"}`}>{pct(a.confidence)}</td>
                    <td className="max-w-[22rem] truncate px-4 text-xs text-muted-foreground">{a.recommended_action}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
