import { useState } from "react";
import { connect, startDemo, type ConsoleState } from "@/lib/quaviz/store";
import { setApiBase } from "@/lib/quaviz/client";

function Dot({ on, warn }: { on: boolean; warn?: boolean }) {
  return <span className={`inline-block h-2 w-2 rounded-full ${on ? "bg-ok" : warn ? "bg-warn" : "bg-alarm"}`} />;
}

export function SystemStrip({ s }: { s: ConsoleState }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(s.base);
  const totalMsgs = Object.values(s.assets).reduce((n, a) => n + (a.messages || 0), 0);
  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-border bg-card px-5 py-2.5 text-sm">
      <div className="flex items-baseline gap-2">
        <span className="text-lg font-bold tracking-wide">QuaViz</span>
        <span className="label-caps">Pump fault console</span>
      </div>
      {s.mode === "demo" && (
        <span className="rounded-sm bg-warn-soft px-2 py-0.5 text-xs font-bold tracking-wider text-warn">DEMO · SYNTHETIC REPLAY</span>
      )}
      <div className="flex items-center gap-2" title={s.error ?? ""}>
        <Dot on={s.healthOk} warn={s.mode === "connecting"} />
        <span>Backend {s.mode === "connecting" ? "connecting…" : s.healthOk ? "connected" : "unreachable"}</span>
      </div>
      <div className="flex items-center gap-2"><Dot on={s.wsConnected} /><span>Live stream {s.wsConnected ? "on" : "off"}</span></div>
      <div className="flex gap-4 font-mono text-xs text-muted-foreground">
        <span>msgs <b className="text-foreground">{totalMsgs.toLocaleString()}</b></span>
        <span>alerts <b className="text-foreground">{s.alertCount}</b></span>
        <span>explanations <b className="text-foreground">{s.explanationCount}</b></span>
        <span>uptime <b className="text-foreground">{Math.round(s.uptime)}s</b></span>
      </div>
      <div className="ml-auto flex items-center gap-2">
        {editing ? (
          <form className="flex gap-1" onSubmit={(e) => { e.preventDefault(); setApiBase(val); setEditing(false); connect(val); }}>
            <input aria-label="Backend URL" value={val} onChange={(e) => setVal(e.target.value)}
              className="w-60 rounded-sm border border-input bg-background px-2 py-1 font-mono text-xs" />
            <button className="rounded-sm bg-primary px-2 py-1 text-xs font-semibold text-primary-foreground">Connect</button>
          </form>
        ) : (
          <button onClick={() => { setVal(s.mode === "demo" ? "" : s.base); setEditing(true); }}
            className="font-mono text-xs text-muted-foreground underline-offset-2 hover:underline">
            {s.mode === "demo" ? "set backend URL" : s.base}
          </button>
        )}
        {s.mode === "demo" ? (
          <button onClick={() => connect()} className="rounded-sm border border-border px-2 py-1 text-xs hover:bg-secondary">Retry backend</button>
        ) : (
          <button onClick={startDemo} className="rounded-sm border border-border px-2 py-1 text-xs hover:bg-secondary">Run demo replay</button>
        )}
      </div>
    </header>
  );
}
