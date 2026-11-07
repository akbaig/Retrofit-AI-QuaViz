import { useEffect, useRef, useState } from "react";
import { humanFault, pct, RED_CONFIDENCE, relTime } from "@/lib/quaviz/status";
import { forceScore } from "@/lib/quaviz/store";
import type { Alert, AssetStatus, Explanation, TileStatus } from "@/lib/quaviz/types";
import { tone } from "./tone";

interface Props {
  id: string;
  status: TileStatus;
  asset?: AssetStatus | undefined;
  alerts: Alert[];
  explanations: Explanation[];
  focusAlertId: string | null;
  live: boolean;
  now: number;
}

function Stat({ k, v }: { k: string; v: string | number }) {
  return <div><div className="label-caps">{k}</div><div className="font-mono text-sm">{v}</div></div>;
}

function cell(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function KvTable({ rows }: { rows: unknown[] }) {
  const objs = rows.filter((r): r is Record<string, unknown> => !!r && typeof r === "object");
  if (!objs.length) return null;
  const cols = [...new Set(objs.flatMap((o) => Object.keys(o)))].slice(0, 5);
  return (
    <table className="w-full text-xs">
      <thead><tr className="label-caps text-left">{cols.map((c) => <th key={c} className="py-1 pr-2 font-semibold">{c}</th>)}</tr></thead>
      <tbody>{objs.slice(0, 8).map((o, i) => (
        <tr key={i} className="border-t border-border/60">{cols.map((c) => <td key={c} className="py-1 pr-2 font-mono">{cell(o[c])}</td>)}</tr>
      ))}</tbody>
    </table>
  );
}

export function AssetDetail({ id, status, asset, alerts, explanations, focusAlertId, live, now }: Props) {
  const t = tone[status];
  const [scoring, setScoring] = useState(false);
  const [scoreErr, setScoreErr] = useState<string | null>(null);
  const focusRef = useRef<HTMLLIElement>(null);
  const active = (focusAlertId && alerts.find((a) => a.id === focusAlertId)) || alerts[0];
  const ex = active && (explanations.find((e) => e.alert_id === active.id) ??
    explanations.find((e) => e.fault_type === active.fault_type));
  const packet = ex?.packet ?? undefined;
  const action = packet?.recommended_action_text || active?.recommended_action;
  const sandbox = ex?.sandbox_action || packet?.sandbox_action;
  const residuals = packet?.top_residuals ?? [];
  const maxRes = Math.max(...residuals.map((r) => Math.abs(r.value)), 1e-9);

  useEffect(() => { focusRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }, [focusAlertId]);

  return (
    <section className="panel flex min-h-0 flex-col overflow-hidden" aria-label={`${id} detail`}>
      <div className={`h-1.5 ${t.bar} transition-colors duration-500`} />
      <header className="flex flex-wrap items-end justify-between gap-3 border-b border-border px-5 py-3">
        <div>
          <div className={`text-xs font-bold tracking-widest ${t.text}`}>{t.label}</div>
          <h2 className="font-mono text-2xl font-semibold">{id}</h2>
        </div>
        <div className="flex gap-5">
          <Stat k="messages" v={(asset?.messages ?? 0).toLocaleString()} />
          <Stat k="minutes" v={asset?.minutes_aggregated ?? 0} />
          <Stat k="alerts" v={asset?.alerts ?? alerts.length} />
          <Stat k="last data" v={relTime(asset?.last_timestamp_utc, now)} />
        </div>
      </header>

      <div className="min-h-0 flex-1 space-y-5 overflow-auto p-5">
        {!active ? (
          <div className="rounded-md border border-dashed border-border p-8 text-center text-muted-foreground">
            No alerts yet — listening to telemetry.
          </div>
        ) : (
          <>
            <div className={`rounded-md border ${active.confidence >= RED_CONFIDENCE ? "border-alarm bg-alarm-soft" : "border-warn bg-warn-soft"} p-4`}>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="label-caps">Active alert · {active.event_class}</div>
                  <div className={`text-2xl font-bold ${active.confidence >= RED_CONFIDENCE ? "text-alarm" : "text-warn"}`}>{humanFault(active.fault_type)}</div>
                  <div className="font-mono text-xs text-muted-foreground">{active.alert_time_utc} · {relTime(active.alert_time_utc, now)}</div>
                </div>
                <div className="text-right">
                  <div className="label-caps">Confidence</div>
                  <div className="font-mono text-2xl font-semibold">{pct(active.confidence)}</div>
                </div>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded-full bg-background/60" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(active.confidence * 100)} aria-label="Confidence">
                <div className={`h-full ${active.confidence >= RED_CONFIDENCE ? "bg-alarm" : "bg-warn"}`} style={{ width: `${active.confidence * 100}%` }} />
              </div>
            </div>

            <div>
              <h3 className="label-caps mb-1.5">What to check</h3>
              <p className="rounded-md border-l-4 border-primary bg-accent px-4 py-3 text-base font-medium text-accent-foreground">
                {action || "No recommended action provided."}
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                {sandbox && <span className="rounded-sm border border-border px-2 py-0.5 font-mono">sandbox measure: {sandbox}</span>}
                <span>Advisory only — human verification on site. This console cannot operate equipment.</span>
              </div>
            </div>

            <div>
              <h3 className="label-caps mb-1.5">Why</h3>
              {active.evidence && <p className="text-sm text-foreground/90">{active.evidence}</p>}
              {ex?.narrative?.summary && (
                <blockquote className="mt-3 border-l-2 border-border pl-3 text-sm leading-relaxed">
                  {ex.narrative.summary}
                  {ex.provider && <footer className="mt-1 font-mono text-xs text-muted-foreground">— {ex.provider}</footer>}
                </blockquote>
              )}
              {!ex && <p className="mt-2 text-xs text-muted-foreground">No narrative explanation for this alert yet.</p>}
              {residuals.length > 0 && (
                <div className="mt-4">
                  <div className="label-caps mb-1">Top residuals</div>
                  <ul className="space-y-1">
                    {residuals.slice(0, 8).map((r) => (
                      <li key={r.name} className="grid grid-cols-[10rem_1fr_4rem] items-center gap-2 text-xs">
                        <span className="truncate font-mono">{r.name}</span>
                        <span className="h-1.5 rounded-full bg-secondary"><span className="block h-full rounded-full bg-primary" style={{ width: `${(Math.abs(r.value) / maxRes) * 100}%` }} /></span>
                        <span className="text-right font-mono">{r.value.toFixed(3)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {packet?.physics && packet.physics.length > 0 && (
                <div className="mt-4"><div className="label-caps mb-1">Physics checks</div><KvTable rows={packet.physics} /></div>
              )}
            </div>
          </>
        )}

        {alerts.length > 0 && (
          <div>
            <h3 className="label-caps mb-1.5">Recent alerts on {id}</h3>
            <ul className="divide-y divide-border/60 text-sm">
              {alerts.slice(0, 15).map((a) => (
                <li key={a.id} ref={a.id === active?.id ? focusRef : undefined}
                  className={`flex justify-between gap-2 py-1.5 ${a.id === active?.id ? "text-foreground" : "text-muted-foreground"}`}>
                  <span className="font-mono text-xs">{relTime(a.alert_time_utc, now)}</span>
                  <span className={a.confidence >= RED_CONFIDENCE ? "text-alarm" : "text-warn"}>{humanFault(a.fault_type)}</span>
                  <span className="font-mono text-xs">{pct(a.confidence)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {live && (
          <div className="border-t border-border pt-3 text-xs text-muted-foreground">
            <button disabled={scoring} onClick={async () => {
              setScoring(true); setScoreErr(null);
              try { await forceScore(id); } catch (e) { setScoreErr((e as Error).message); } finally { setScoring(false); }
            }} className="rounded-sm border border-border px-2 py-1 hover:bg-secondary disabled:opacity-50">
              {scoring ? "Re-running analysis…" : "Re-run fault analysis"}
            </button>
            <span className="ml-2">Re-scores telemetry in the backend; no effect on the pump.</span>
            {scoreErr && <div className="mt-1 text-alarm">{scoreErr}</div>}
          </div>
        )}
      </div>
    </section>
  );
}
