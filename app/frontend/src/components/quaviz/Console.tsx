import { useEffect, useState } from "react";
import { connect, startDemo, useConsole } from "@/lib/quaviz/store";
import { pumpIds, tileStatus } from "@/lib/quaviz/status";
import type { Alert } from "@/lib/quaviz/types";
import { AssetDetail } from "./AssetDetail";
import { ContextBar } from "./ContextBar";
import { LiveLog } from "./LiveLog";
import { PumpTile } from "./PumpTile";
import { StatsRail } from "./StatsRail";

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

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelected(null);
        setFocusAlert(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const reachable = s.mode === "live" || s.mode === "demo" || s.mode === "connecting";
  const ids = pumpIds(s.assets, s.alerts);
  const byAsset = (id: string) => s.alerts.filter((a) => a.asset_id === id);
  const counts = { red: 0, amber: 0, green: 0, grey: 0 };
  ids.forEach((id) => counts[tileStatus(s.assets[id], byAsset(id), reachable)]++);

  const openPump = (id: string, alert?: Alert | null) => {
    setSelected(id);
    setFocusAlert(alert?.id ?? null);
  };

  return (
    <div className="flex min-h-screen flex-col lg:h-screen lg:overflow-hidden">
      <ContextBar s={s} counts={counts} />

      {s.mode === "offline" && (
        <div role="alert" className="flex flex-wrap items-center gap-3 border-b border-warn/40 bg-warn-soft px-5 py-2 text-sm">
          <span className="font-semibold text-warn">Backend unreachable at {s.base}.</span>
          <span className="text-muted-foreground">
            Run <code className="font-mono">./start.sh</code> from the repo root, or use the demo.
          </span>
          <button
            type="button"
            onClick={startDemo}
            className="ml-auto rounded-sm bg-primary px-3 py-1 text-xs font-semibold text-primary-foreground"
          >
            Run in-browser demo replay
          </button>
        </div>
      )}

      <div className="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[14rem_minmax(0,1fr)_18rem]">
        <StatsRail s={s} />

        <main className="flex min-h-0 flex-col" aria-label="Fleet wall">
          <div className="mb-2 flex items-center justify-between px-1">
            <h2 className="label-caps">Fleet · {ids.length} pumps</h2>
            <span className="text-xs text-muted-foreground">Select a pump for detail</span>
          </div>
          <div className="min-h-0 flex-1 overflow-auto">
            {s.loading ? (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
                {ids.map((id) => (
                  <div key={id} className="h-28 animate-pulse rounded-md bg-muted" />
                ))}
              </div>
            ) : (
              <div className="mx-auto grid max-w-5xl grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
                {ids.map((id) => {
                  const al = byAsset(id);
                  return (
                    <PumpTile
                      key={id}
                      id={id}
                      status={tileStatus(s.assets[id], al, reachable)}
                      asset={s.assets[id]}
                      latest={al[0]}
                      selected={id === selected}
                      pulseKey={s.pulse[id]}
                      now={now}
                      onSelect={() => openPump(id)}
                    />
                  );
                })}
              </div>
            )}
          </div>
        </main>

        <LiveLog
          entries={s.log}
          onPickAsset={(id) => openPump(id)}
        />
      </div>

      {/* Detail side pane */}
      {selected && (
        <>
          <button
            type="button"
            aria-label="Close detail"
            className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[1px] transition-opacity"
            onClick={() => {
              setSelected(null);
              setFocusAlert(null);
            }}
          />
          <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl animate-row-in flex-col p-3 sm:p-4">
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-md shadow-2xl shadow-black/40">
              <AssetDetail
                id={selected}
                status={tileStatus(s.assets[selected], byAsset(selected), reachable)}
                asset={s.assets[selected]}
                alerts={byAsset(selected)}
                explanations={s.explanations.filter((e) => e.asset_id === selected)}
                focusAlertId={focusAlert}
                live={s.mode === "live"}
                now={now}
                onClose={() => {
                  setSelected(null);
                  setFocusAlert(null);
                }}
              />
            </div>
          </div>
        </>
      )}

      {s.lastErrorEvent && (
        <div className="border-t border-border px-5 py-1 text-xs text-alarm">
          Backend reported: {s.lastErrorEvent}
        </div>
      )}
    </div>
  );
}
