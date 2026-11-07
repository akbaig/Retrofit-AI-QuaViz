export interface AssetStatus {
  asset_id: string;
  messages: number;
  minutes_aggregated: number;
  feature_rows: number;
  alerts: number;
  last_timestamp_utc: string | null;
  last_infer_utc: string | null;
}

export interface Alert {
  id: string;
  asset_id: string;
  alert_time_utc: string;
  event_class: string;
  fault_type: string;
  confidence: number;
  evidence?: string | null;
  recommended_action?: string | null;
  received_at: number;
}

export interface Residual {
  name: string;
  value: number;
}

export interface ExplainPacket {
  confidence?: number;
  recommended_action_text?: string;
  top_residuals?: Residual[];
  physics?: unknown[];
  snapshots?: unknown[];
  context?: Record<string, unknown>;
  sandbox_action?: string;
}

export interface Explanation {
  id: string;
  asset_id: string;
  alert_id?: string | null;
  alert_time_utc?: string;
  fault_type?: string;
  narrative?: { summary?: string; [k: string]: unknown } | null;
  packet?: ExplainPacket | null;
  sandbox_action?: string | null;
  provider?: string | null;
  what_if?: unknown;
  received_at?: number;
}

export interface StatusSnapshot {
  uptime_s: number;
  assets: Record<string, AssetStatus>;
  alert_count: number;
  explanation_count: number;
}

export interface Health {
  ok: boolean;
  model?: string;
  data?: string;
  explain?: boolean;
  infer_every?: number;
  mqtt_host?: string;
}

export type TileStatus = "red" | "amber" | "green" | "grey";
