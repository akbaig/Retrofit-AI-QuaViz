import type { Alert, AssetStatus, Explanation, Health, StatusSnapshot } from "./types";

const KEY = "quaviz.apiBase";
export const DEFAULT_BASE = (import.meta.env["VITE_API_BASE"] as string | undefined) ?? "http://127.0.0.1:8000";

export function getApiBase(): string {
  if (typeof window === "undefined") return DEFAULT_BASE;
  return window.localStorage.getItem(KEY) || DEFAULT_BASE;
}
export function setApiBase(v: string) {
  window.localStorage.setItem(KEY, v.replace(/\/+$/, ""));
}
export function wsUrl(base: string): string {
  return base.replace(/^http(s?):/i, (_m, s: string) => `ws${s}:`).replace(/\/+$/, "") + "/ws/events";
}

async function get<T>(base: string, path: string): Promise<T> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 4000);
  try {
    const r = await fetch(base + path, { signal: ctrl.signal });
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return (await r.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

export const api = {
  health: (b: string) => get<Health>(b, "/health"),
  status: (b: string) => get<StatusSnapshot>(b, "/status"),
  assets: (b: string) => get<{ assets: AssetStatus[] }>(b, "/assets").then((r) => r.assets ?? []),
  alerts: (b: string) => get<{ alerts: Alert[] }>(b, "/alerts?limit=500").then((r) => r.alerts ?? []),
  explanations: (b: string) =>
    get<{ explanations: Explanation[] }>(b, "/explanations?limit=200").then((r) => r.explanations ?? []),
  score: async (b: string, id: string) => {
    const r = await fetch(`${b}/score/${encodeURIComponent(id)}`, { method: "POST" });
    if (!r.ok) throw new Error(`score: HTTP ${r.status}`);
    return (await r.json()) as { alerts: Alert[] };
  },
};

export type WsEvent =
  | ({ type: "alert" } & Alert)
  | ({ type: "explain" } & Explanation)
  | { type: "error"; message?: string; [k: string]: unknown };

/** Parse a WS frame defensively; returns null for anything malformed. */
export function parseEvent(raw: unknown): WsEvent | null {
  if (typeof raw !== "string") return null;
  try {
    const o = JSON.parse(raw) as Record<string, unknown>;
    if (!o || typeof o !== "object") return null;
    if (o["type"] === "alert" && typeof o["asset_id"] === "string" && typeof o["confidence"] === "number") return o as WsEvent;
    if (o["type"] === "explain" && typeof o["asset_id"] === "string") return o as WsEvent;
    if (o["type"] === "error") return o as WsEvent;
    return null;
  } catch {
    return null;
  }
}
