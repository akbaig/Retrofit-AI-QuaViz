import { useEffect, useState } from "react";
import { connect, startDemo, useConsole } from "@/lib/quaviz/store";
import { pumpIds, tileStatus } from "@/lib/quaviz/status";
import type { Alert } from "@/lib/quaviz/types";
import { AlertStream } from "./AlertStream";
import { AssetDetail } from "./AssetDetail";
import { PumpTile } from "./PumpTile";
import { SystemStrip } from "./SystemStrip";

export function Console() {
  const s = useConsole();
  const [selected, setSelected] = useState<string | null>(null);
  const [focusAlert, setFocusAlert] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    connect();
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  const reachable = s.mode === "live" || s.mode === "demo" || s.mode === "connecting";
  const ids = pumpIds(s.assets, s.alerts);
  const byAsset = (id: string) => s.alerts.filter((a) => a.asset_id === id);
  const sel = selected ?? ids[0];
  const counts = { red: 0, amber: 0, green: 0, grey: 0 };
  ids.forEach((id) => counts[tileStatus(s.assets[id], byAsset(id), reachable)]++);

  const pick = (a: Alert) => { setSelected(a.asset_id); setFocusAlert(a.id); };

  return (
    <div className="flex min-h-screen flex-col lg:h-screen">
      <SystemStrip s={s} />
      {s.mode === "offline" && (
        <div role="alert" className="flex flex-wrap items-center gap-3 border-b border-warn/40 bg-warn-soft px-5 py-2 text-sm">
          <span className="font-semibold text-warn">Backend unreachable at {s.base}.</span>
          <span className="text-muted-foreground">
            Start it from the repo root with <code className="font-mono">./scripts/start_mqtt_demo.sh</code>, or set a different URL. Retrying every 7s.
          </span>
          <button onClick={startDemo} className="ml-auto rounded-sm bg-primary px-3 py-1 text-xs font-semibold text-primary-foreground">
            Run in-browser demo replay
          </button>
        </div>
      )}
      <main className="grid min-h-0 flex-1 gap-4 p-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div className="flex min-h-0 flex-col gap-4">
          <section aria-label="Fleet wall">
            <div className="mb-2 flex items-center justify-between">
              <h1 className="label-caps">Fleet wall · {ids.length} pumps</h1>
              <div className="flex gap-3 font-mono text-xs">
                <span className="text-alarm">● {counts.red} fault</span>
                <span className="text-warn">● {counts.amber} watch</span>
                <span className="text-ok">● {counts.green} healthy</span>
                <span className="text-muted-foreground">● {counts.grey} no data</span>
              </div>
            </div>
            {s.loading ? (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
                {ids.map((id) => <div key={id} className="h-28 animate-pulse rounded-md bg-muted" />)}
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
                {ids.map((id) => {
                  const al = byAsset(id);
                  return (
                    <PumpTile key={id} id={id} status={tileStatus(s.assets[id], al, reachable)} asset={s.assets[id]}
                      latest={al[0]} selected={id === sel} pulseKey={s.pulse[id]} now={now}
                      onSelect={() => { setSelected(id); setFocusAlert(null); }} />
                  );
                })}
              </div>
            )}
          </section>
          <div className="flex min-h-[18rem] flex-1 flex-col">
            <AlertStream alerts={s.alerts} filter={selected} fresh={s.freshIds} onPick={pick}
              onClearFilter={() => setSelected(null)} />
          </div>
        </div>
        {sel && (
          <AssetDetail id={sel} status={tileStatus(s.assets[sel], byAsset(sel), reachable)} asset={s.assets[sel]}
            alerts={byAsset(sel)} explanations={s.explanations.filter((e) => e.asset_id === sel)}
            focusAlertId={focusAlert} live={s.mode === "live"} now={now} />
        )}
      </main>
      {s.lastErrorEvent && <div className="border-t border-border px-5 py-1 text-xs text-alarm">Backend reported: {s.lastErrorEvent}</div>}
    </div>
  );
}
