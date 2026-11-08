import type { WsEvent } from "./client";
import { DEFAULT_PUMPS } from "./status";
import type { AssetStatus } from "./types";

const FAULTS = [
  { fault: "BEARING_WEAR", cls: "MACHINE", action: "Inspect bearing lubrication and condition; confirm with vibration measurements.",
    residuals: ["vib_rms_de", "bearing_temp_c", "vib_kurtosis"], sandbox: "inspect" },
  { fault: "CAVITATION", cls: "PROCESS", action: "Check suction pressure and NPSH margin; verify inlet strainer is not blocked.",
    residuals: ["suction_kpa", "flow_m3h", "vib_hf_band"], sandbox: "observe" },
  { fault: "SEAL_LEAK", cls: "MACHINE", action: "Visually inspect mechanical seal area; check seal flush and drain.",
    residuals: ["seal_chamber_kpa", "motor_current_a", "flow_m3h"], sandbox: "inspect" },
  { fault: "IMPELLER_FOULING", cls: "MACHINE", action: "Compare head vs. flow against curve; schedule impeller inspection at next stop.",
    residuals: ["head_m", "efficiency", "motor_current_a"], sandbox: "schedule" },
  { fault: "SENSOR_DRIFT", cls: "SENSOR", action: "Cross-check the flagged sensor against a handheld reference before acting.",
    residuals: ["discharge_kpa", "flow_m3h"], sandbox: "verify" },
];

const uid = () => (globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2));

export function createDemoFeed(h: {
  onAssets: (a: Record<string, AssetStatus>, uptime: number) => void;
  onEvent: (e: WsEvent) => void;
}) {
  const start = Date.now();
  const assets: Record<string, AssetStatus> = {};
  // Pumps come online one by one; PUMP-12 stays silent to show the "never seen" state.
  const online = DEFAULT_PUMPS.slice(0, 11);
  const tick = setInterval(() => {
    const now = new Date();
    const age = (Date.now() - start) / 1000;
    online.forEach((id, i) => {
      if (age < i * 0.6) return;
      const a = (assets[id] ??= { asset_id: id, messages: 0, minutes_aggregated: 0, feature_rows: 0, alerts: 0,
        last_timestamp_utc: null, last_infer_utc: null });
      a.messages += 3 + Math.floor(Math.random() * 6);
      a.minutes_aggregated = Math.floor(a.messages / 60);
      a.feature_rows = a.minutes_aggregated;
      a.last_timestamp_utc = now.toISOString();
      if (a.minutes_aggregated >= 1 && i !== 10) a.last_infer_utc = now.toISOString();
    });
    h.onAssets(structuredClone(assets), Math.round(age));
  }, 1000);

  const fire = () => {
    const ids = Object.keys(assets).filter((id) => assets[id]?.last_infer_utc);
    if (!ids.length) return;
    const id = ids[Math.floor(Math.random() * Math.min(ids.length, 6))]!;
    const f = FAULTS[Math.floor(Math.random() * FAULTS.length)]!;
    const conf = Math.random() < 0.65 ? 0.82 + Math.random() * 0.17 : 0.55 + Math.random() * 0.24;
    const alertId = uid();
    const time = new Date().toISOString();
    assets[id]!.alerts++;
    h.onEvent({ type: "alert", id: alertId, asset_id: id, alert_time_utc: time, event_class: f.cls, fault_type: f.fault,
      confidence: conf, evidence: `${f.residuals[0]} residual exceeded baseline by ${(2 + Math.random() * 3).toFixed(1)}σ over the last 3 aggregated minutes.`,
      recommended_action: f.action, received_at: Date.now() / 1000 });
    setTimeout(() => {
      h.onEvent({ type: "explain", id: uid(), asset_id: id, alert_id: alertId, alert_time_utc: time, fault_type: f.fault,
        narrative: { summary: `${id} shows a pattern consistent with ${f.fault.toLowerCase().replace(/_/g, " ")}. The strongest deviation is in ${f.residuals[0]}, supported by ${f.residuals.slice(1).join(" and ")}. Confirm on site before any intervention.` },
        packet: { confidence: conf, recommended_action_text: f.action,
          top_residuals: f.residuals.map((n, k) => ({ name: n, value: +(0.4 / (k + 1) + Math.random() * 0.1).toFixed(3) })),
          physics: [{ check: "affinity law head/flow", status: Math.random() > 0.5 ? "violated" : "within band" }],
          snapshots: [], context: {}, sandbox_action: f.sandbox },
        sandbox_action: f.sandbox, provider: "TemplateExplainer (demo)", what_if: null });
    }, 1500);
  };
  const first = setTimeout(fire, 6000);
  const loop = setInterval(fire, 9000);
  return () => { clearInterval(tick); clearInterval(loop); clearTimeout(first); };
}
