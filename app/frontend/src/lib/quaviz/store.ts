import { useSyncExternalStore } from "react";
import { api, getApiBase, parseEvent, wsUrl, type WsEvent } from "./client";
import { createDemoFeed } from "./demo";
import { ensureNotifyPermission, notifyAlert, notifyExplain } from "./notify";
import { humanFault, pct, RED_CONFIDENCE } from "./status";
import type { Alert, AssetStatus, Explanation, FleetContext, Health, LogEntry } from "./types";

export type Mode = "connecting" | "live" | "offline" | "demo";

export interface ConsoleState {
  base: string;
  mode: Mode;
  health: Health | null;
  healthOk: boolean;
  wsConnected: boolean;
  loading: boolean;
  error: string | null;
  uptime: number;
  alertCount: number;
  explanationCount: number;
  assets: Record<string, AssetStatus>;
  alerts: Alert[];
  explanations: Explanation[];
  context: FleetContext | null;
  log: LogEntry[];
  pulse: Record<string, number>;
  freshIds: Record<string, true>;
  lastErrorEvent: string | null;
}

let state: ConsoleState = {
  base: "", mode: "connecting", health: null, healthOk: false, wsConnected: false, loading: true,
  error: null, uptime: 0, alertCount: 0, explanationCount: 0, assets: {}, alerts: [], explanations: [],
  context: null, log: [], pulse: {}, freshIds: {}, lastErrorEvent: null,
};
const listeners = new Set<() => void>();
const set = (p: Partial<ConsoleState>) => {
  state = { ...state, ...p };
  listeners.forEach((l) => l());
};
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};
export const useConsole = () => useSyncExternalStore(subscribe, () => state, () => state);
export const getState = () => state;

let logSeq = 0;
function pushLog(entry: Omit<LogEntry, "id" | "at"> & { at?: number }) {
  const row: LogEntry = {
    id: `log-${++logSeq}`,
    at: entry.at ?? Date.now(),
    kind: entry.kind,
    title: entry.title,
    ...(entry.asset_id !== undefined ? { asset_id: entry.asset_id } : {}),
    ...(entry.detail !== undefined ? { detail: entry.detail } : {}),
  };
  set({ log: [row, ...state.log].slice(0, 400) });
}

export function ingest(ev: WsEvent, opts?: { silent?: boolean }) {
  if (ev.type === "alert") {
    const { type: _t, ...a } = ev;
    if (state.alerts.some((x) => x.id === a.id)) return;
    const prev = state.assets[a.asset_id];
    const assets = {
      ...state.assets,
      [a.asset_id]: prev ? { ...prev, alerts: prev.alerts + 1 } : {
        asset_id: a.asset_id, messages: 0, minutes_aggregated: 0, feature_rows: 0, alerts: 1,
        last_timestamp_utc: a.alert_time_utc, last_infer_utc: null,
      },
    };
    set({
      alerts: [a, ...state.alerts].slice(0, 500),
      assets,
      alertCount: state.alertCount + 1,
      pulse: a.confidence >= RED_CONFIDENCE ? { ...state.pulse, [a.asset_id]: Date.now() } : state.pulse,
      freshIds: { ...state.freshIds, [a.id]: true },
    });
    pushLog({
      kind: "alert",
      asset_id: a.asset_id,
      title: `${a.asset_id} · ${humanFault(a.fault_type)}`,
      detail: `${pct(a.confidence)} · ${a.recommended_action ?? "check evidence"}`,
    });
    if (!opts?.silent) notifyAlert(a);
  } else if (ev.type === "explain") {
    const { type: _t, ...e } = ev;
    if (state.explanations.some((x) => x.id === e.id)) return;
    set({ explanations: [e, ...state.explanations].slice(0, 200), explanationCount: state.explanationCount + 1 });
    const summary = e.narrative?.summary ? String(e.narrative.summary).slice(0, 140) : "explanation ready";
    pushLog({
      kind: "explain",
      asset_id: e.asset_id,
      title: `${e.asset_id} · explain`,
      detail: summary,
    });
    if (!opts?.silent) notifyExplain(e);
  } else {
    const msg = String(ev.message ?? ev["error"] ?? "backend error event");
    set({ lastErrorEvent: msg });
    pushLog({ kind: "error", title: "Backend error", detail: msg });
  }
}

let ws: WebSocket | null = null;
let poll: ReturnType<typeof setInterval> | null = null;
let retry: ReturnType<typeof setTimeout> | null = null;
let demoStop: (() => void) | null = null;
let gen = 0;
let lastMsgFingerprint = "";

function teardown() {
  gen++;
  ws?.close();
  ws = null;
  if (poll) clearInterval(poll);
  if (retry) clearTimeout(retry);
  demoStop?.();
  demoStop = null;
}

function noteTelemetry(assets: Record<string, AssetStatus>) {
  // Log on minute progress (not every MQTT message burst).
  const parts = Object.values(assets)
    .filter((a) => a.minutes_aggregated > 0 || a.messages > 0)
    .sort((a, b) => a.asset_id.localeCompare(b.asset_id))
    .map((a) => `${a.asset_id}:${a.minutes_aggregated}`);
  const fp = parts.join("|");
  if (!fp || fp === lastMsgFingerprint) return;
  const grew = Object.values(assets).filter((a) => a.messages > 0);
  const totalMsgs = grew.reduce((n, a) => n + a.messages, 0);
  const totalMin = grew.reduce((n, a) => n + a.minutes_aggregated, 0);
  lastMsgFingerprint = fp;
  pushLog({
    kind: "telemetry",
    title: `Telemetry · ${grew.length} pumps`,
    detail: `${totalMsgs.toLocaleString()} msgs · ${totalMin} aggregated minutes`,
  });
}

async function bootstrap(base: string, g: number): Promise<boolean> {
  const [health, status, assets, alerts, explanations, context] = await Promise.all([
    api.health(base), api.status(base), api.assets(base), api.alerts(base), api.explanations(base),
    api.context(base).catch(() => null),
  ]);
  if (g !== gen) return false;
  const map: Record<string, AssetStatus> = { ...status.assets };
  assets.forEach((a) => (map[a.asset_id] = a));
  set({
    health, healthOk: !!health.ok, mode: "live", loading: false, assets: map, alerts, explanations,
    context, uptime: status.uptime_s, alertCount: status.alert_count, explanationCount: status.explanation_count,
    error: null,
  });
  noteTelemetry(map);
  pushLog({ kind: "system", title: "Connected to backend", detail: base });
  return true;
}

async function refreshStatus(base: string, g: number) {
  const wasOffline = state.mode === "offline";
  try {
    if (wasOffline) {
      await bootstrap(base, g);
      return;
    }
    const [s, context] = await Promise.all([
      api.status(base),
      api.context(base).catch(() => state.context),
    ]);
    if (g !== gen) return;
    const assets = { ...state.assets, ...s.assets };
    set({
      assets, uptime: s.uptime_s, alertCount: s.alert_count,
      explanationCount: s.explanation_count, healthOk: true, mode: "live", error: null, context,
    });
    noteTelemetry(assets);
  } catch (e) {
    if (g !== gen) return;
    set({ healthOk: false, mode: "offline", error: (e as Error).message });
  }
}

function openWs(base: string, g: number) {
  if (g !== gen) return;
  try {
    ws = new WebSocket(wsUrl(base));
  } catch {
    retry = setTimeout(() => openWs(base, g), 3000);
    return;
  }
  ws.onopen = () => g === gen && set({ wsConnected: true });
  ws.onmessage = (m) => {
    const ev = parseEvent(m.data);
    if (ev && g === gen) ingest(ev);
  };
  ws.onclose = () => {
    if (g !== gen) return;
    set({ wsConnected: false });
    retry = setTimeout(() => openWs(base, g), 3000);
  };
  ws.onerror = () => ws?.close();
}

export async function connect(base = getApiBase()) {
  teardown();
  lastMsgFingerprint = "";
  const g = gen;
  set({
    base, mode: "connecting", loading: true, error: null, wsConnected: false,
    pulse: {}, freshIds: {}, log: [], context: null,
  });
  ensureNotifyPermission();
  try {
    await bootstrap(base, g);
  } catch (e) {
    if (g !== gen) return;
    set({ mode: "offline", loading: false, healthOk: false, error: (e as Error).message });
    pushLog({ kind: "error", title: "Backend unreachable", detail: (e as Error).message });
  }
  openWs(base, g);
  poll = setInterval(() => {
    refreshStatus(base, g);
    api.health(base).then((h) => g === gen && set({ health: h, healthOk: !!h.ok })).catch(() => undefined);
  }, 5000);
}

export function startDemo() {
  teardown();
  lastMsgFingerprint = "";
  set({
    base: "in-browser synthetic replay", mode: "demo", loading: false, error: null, healthOk: true,
    wsConnected: true, health: { ok: true, explain: true, mqtt_host: "demo" }, assets: {}, alerts: [],
    explanations: [], alertCount: 0, explanationCount: 0, pulse: {}, freshIds: {}, uptime: 0, log: [],
    context: {
      ok: true,
      fleet: {
        n_assets: 11,
        schedule_state: "sea passage",
        duty_demand_pct: 86,
        sea_temp_c: 18.4,
        supply_water_temp_c: 22.1,
        ambient_temp_c: 34.2,
        sea_state_hs_m: 1.6,
      },
      assets: {},
    },
  });
  pushLog({ kind: "system", title: "Demo replay started", detail: "Synthetic fleet — advisory only" });
  demoStop = createDemoFeed({
    onAssets: (assets, uptime) => {
      set({ assets, uptime });
      noteTelemetry(assets);
      // Drift demo context slightly
      const f = state.context?.fleet;
      if (f) {
        set({
          context: {
            ok: true,
            fleet: {
              ...f,
              sea_temp_c: +(18.2 + Math.sin(uptime / 40) * 0.6).toFixed(1),
              ambient_temp_c: +(33.5 + Math.sin(uptime / 55) * 1.2).toFixed(1),
              duty_demand_pct: +(84 + Math.sin(uptime / 30) * 6).toFixed(1),
              sea_state_hs_m: +(1.4 + Math.abs(Math.sin(uptime / 70)) * 0.8).toFixed(1),
            },
            assets: {},
          },
        });
      }
    },
    onEvent: ingest,
  });
}

export async function forceScore(id: string) {
  if (state.mode !== "live") return;
  const r = await api.score(state.base, id);
  r.alerts?.forEach((a) => ingest({ type: "alert", ...a }, { silent: true }));
}
