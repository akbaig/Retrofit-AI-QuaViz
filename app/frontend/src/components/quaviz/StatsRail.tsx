import {
  AlertTriangle,
  Clock,
  Link2,
  MessageSquare,
  Radio,
  Server,
  Sparkles,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { setApiBase } from "@/lib/quaviz/client";
import { connect, startDemo, type ConsoleState } from "@/lib/quaviz/store";

function Dot({ on, warn }: { on: boolean; warn?: boolean }) {
  return (
    <span
      className={`inline-block h-2 w-2 shrink-0 rounded-full ${on ? "bg-ok animate-pulse" : warn ? "bg-warn" : "bg-alarm"}`}
    />
  );
}

function Row({
  label,
  icon,
  children,
}: {
  label: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="border-b border-border/50 py-2.5">
      <div className="mb-1 flex items-center gap-1.5 text-muted-foreground">
        <span className="opacity-80">{icon}</span>
        <span className="label-caps !mb-0">{label}</span>
      </div>
      <div className="text-sm">{children}</div>
    </div>
  );
}

export function StatsRail({ s }: { s: ConsoleState }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(s.base);
  const totalMsgs = Object.values(s.assets).reduce((n, a) => n + (a.messages || 0), 0);
  const backendLabel =
    s.mode === "connecting" ? "Connecting…" : s.healthOk ? "Connected" : "Unreachable";
  const ic = "h-3.5 w-3.5";

  return (
    <aside className="panel flex h-full min-h-0 flex-col overflow-hidden" aria-label="System stats">
      <div className="flex items-center gap-2 border-b border-border px-4 py-2">
        <Server className="h-3.5 w-3.5 text-primary" aria-hidden />
        <h2 className="label-caps !mb-0">System</h2>
      </div>
      <div className="min-h-0 flex-1 overflow-auto px-4 py-1">
        <Row label="Backend" icon={<Server className={ic} />}>
          <div className="flex items-center gap-2" title={s.error ?? ""}>
            <Dot on={s.healthOk} warn={s.mode === "connecting"} />
            <span className={s.healthOk ? "text-ok" : "text-alarm"}>{backendLabel}</span>
          </div>
        </Row>
        <Row label="Live stream" icon={<Radio className={ic} />}>
          <div className="flex items-center gap-2">
            <Dot on={s.wsConnected} />
            <span>{s.wsConnected ? "On" : "Off"}</span>
          </div>
        </Row>
        <Row label="Messages" icon={<MessageSquare className={ic} />}>
          <span className="font-mono text-base font-semibold tabular-nums">{totalMsgs.toLocaleString()}</span>
        </Row>
        <Row label="Alerts" icon={<AlertTriangle className={ic} />}>
          <span className="font-mono text-base font-semibold tabular-nums">{s.alertCount}</span>
        </Row>
        <Row label="Explanations" icon={<Sparkles className={ic} />}>
          <span className="font-mono text-base font-semibold tabular-nums">{s.explanationCount}</span>
        </Row>
        <Row label="Uptime" icon={<Clock className={ic} />}>
          <span className="font-mono text-base font-semibold tabular-nums">{Math.round(s.uptime)}s</span>
        </Row>
        <Row label="API" icon={<Link2 className={ic} />}>
          {editing ? (
            <form
              className="flex flex-col gap-1"
              onSubmit={(e) => {
                e.preventDefault();
                setApiBase(val);
                setEditing(false);
                connect(val);
              }}
            >
              <input
                aria-label="Backend URL"
                value={val}
                onChange={(e) => setVal(e.target.value)}
                className="w-full rounded-sm border border-input bg-background px-2 py-1 font-mono text-xs"
              />
              <button type="submit" className="rounded-sm bg-primary px-2 py-1 text-xs font-semibold text-primary-foreground">
                Connect
              </button>
            </form>
          ) : (
            <button
              type="button"
              onClick={() => {
                setVal(s.mode === "demo" ? "http://127.0.0.1:8000" : s.base);
                setEditing(true);
              }}
              className="break-all text-left font-mono text-xs text-muted-foreground underline-offset-2 hover:underline"
            >
              {s.mode === "demo" ? "set backend URL" : s.base || "—"}
            </button>
          )}
        </Row>
      </div>
      <div className="border-t border-border p-3">
        {s.mode === "demo" ? (
          <button
            type="button"
            onClick={() => connect()}
            className="w-full rounded-sm border border-border px-3 py-2 text-xs font-semibold hover:bg-secondary"
          >
            Retry backend
          </button>
        ) : (
          <button
            type="button"
            onClick={startDemo}
            className="inline-flex w-full items-center justify-center gap-2 rounded-sm border border-border px-3 py-2 text-xs font-semibold hover:bg-secondary"
          >
            <Radio className="h-3.5 w-3.5" aria-hidden />
            Run demo replay
          </button>
        )}
      </div>
    </aside>
  );
}
