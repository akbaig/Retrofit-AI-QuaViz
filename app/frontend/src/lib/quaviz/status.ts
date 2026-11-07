import type { Alert, AssetStatus, TileStatus } from "./types";

export const RED_CONFIDENCE = 0.8;

export const DEFAULT_PUMPS = Array.from({ length: 12 }, (_, i) => `PUMP-${String(i + 1).padStart(2, "0")}`);

/**
 * Tile colour rules (from the operator brief):
 * - backend unreachable → amber
 * - any alert with confidence >= 0.8 → red
 * - alerts only below 0.8 → amber
 * - never seen (messages == 0 / unknown) → grey
 * - receiving data but not yet inferred → amber
 * - inferred, no alerts → green
 */
export function tileStatus(
  asset: AssetStatus | undefined,
  alerts: Alert[],
  backendReachable: boolean,
): TileStatus {
  if (!backendReachable) return "amber";
  if (alerts.some((a) => a.confidence >= RED_CONFIDENCE)) return "red";
  if (alerts.length > 0) return "amber";
  if (!asset || asset.messages === 0) return "grey";
  if (!asset.last_infer_utc) return "amber";
  return "green";
}

export function pumpIds(assets: Record<string, AssetStatus>, alerts: Alert[]): string[] {
  const ids = new Set<string>([...DEFAULT_PUMPS, ...Object.keys(assets), ...alerts.map((a) => a.asset_id)]);
  return [...ids].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

export function relTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const s = Math.round((now - t) / 1000);
  if (s < 0) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export const pct = (c: number) => `${Math.round(c * 1000) / 10}%`;
export const humanFault = (f?: string | null) => (f ? f.replace(/_/g, " ") : "—");
