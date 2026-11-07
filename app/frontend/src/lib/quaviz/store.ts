import { useSyncExternalStore } from "react";
import { api, getApiBase, parseEvent, wsUrl, type WsEvent } from "./client";
import { createDemoFeed } from "./demo";
import { RED_CONFIDENCE } from "./status";
import type { Alert, AssetStatus, Explanation, Health } from "./types";

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
  pulse: Record<string, number>;
  freshIds: Record<string, true>;
  lastErrorEvent: string | null;
}

let state: ConsoleState = {
  base: "", mode: "connecting", health: null, healthOk: false, wsConnected: false, loading: true,
  error: null, uptime: 0, alertCount: 0, explanationCount: 0, assets: {}, alerts: [], explanations: [],
  pulse: {}, freshIds: {}, lastErrorEvent: null,
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

export function ingest(ev: WsEvent) {
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
  } else if (ev.type === "explain") {
    const { type: _t, ...e } = ev;
    if (state.explanations.some((x) => x.id === e.id)) return;
    set({ explanations: [e, ...state.explanations].slice(0, 200), explanationCount: state.explanationCount + 1 });
  } else {
    set({ lastErrorEvent: String(ev.message ?? ev["error"] ?? "backend error event") });
  }
}

let ws: WebSocket | null = null;
let poll: ReturnType<typeof setInterval> | null = null;
let retry: ReturnType<typeof setTimeout> | null = null;
let demoStop: (() => void) | null = null;
let gen = 0;

function teardown() {
  gen++;
  ws?.close();
  ws = null;
  if (poll) clearInterval(poll);
  if (retry) clearTimeout(retry);
  demoStop?.();
  demoStop = null;
}

async function refreshStatus(base: string, g: number) {
  try {
    const s = await api.status(base);
    if (g !== gen) return;
    set({ assets: { ...state.assets, ...s.assets }, uptime: s.uptime_s, alertCount: s.alert_count,
      explanationCount: s.explanation_count, healthOk: true, mode: "live", error: null });
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
  const g = gen;
  set({ base, mode: "connecting", loading: true, error: null, wsConnected: false, pulse: {}, freshIds: {} });
  try {
    const [health, status, assets, alerts, explanations] = await Promise.all([
      api.health(base), api.status(base), api.assets(base), api.alerts(base), api.explanations(base),
    ]);
    if (g !== gen) return;
    const map: Record<string, AssetStatus> = { ...status.assets };
    assets.forEach((a) => (map[a.asset_id] = a));
    set({ health, healthOk: !!health.ok, mode: "live", loading: false, assets: map, alerts, explanations,
      uptime: status.uptime_s, alertCount: status.alert_count, explanationCount: status.explanation_count });
  } catch (e) {
    if (g !== gen) return;
    set({ mode: "offline", loading: false, healthOk: false, error: (e as Error).message });
  }
  openWs(base, g);
  poll = setInterval(() => {
    refreshStatus(base, g);
    api.health(base).then((h) => g === gen && set({ health: h, healthOk: !!h.ok })).catch(() => undefined);
  }, 7000);
}

export function startDemo() {
  teardown();
  set({ base: "in-browser synthetic replay", mode: "demo", loading: false, error: null, healthOk: true,
    wsConnected: true, health: { ok: true, explain: true, mqtt_host: "demo" }, assets: {}, alerts: [],
    explanations: [], alertCount: 0, explanationCount: 0, pulse: {}, freshIds: {}, uptime: 0 });
  demoStop = createDemoFeed({
    onAssets: (assets, uptime) => set({ assets, uptime }),
    onEvent: ingest,
  });
}

export async function forceScore(id: string) {
  if (state.mode !== "live") return;
  const r = await api.score(state.base, id);
  r.alerts?.forEach((a) => ingest({ type: "alert", ...a }));
}
