import { describe, expect, it } from "vitest";
import { tileStatus, pumpIds } from "./status";
import type { Alert, AssetStatus } from "./types";

const asset = (p: Partial<AssetStatus> = {}): AssetStatus => ({
  asset_id: "PUMP-08", messages: 100, minutes_aggregated: 5, feature_rows: 5, alerts: 0,
  last_timestamp_utc: null, last_infer_utc: "2025-09-10T09:52:00Z", ...p,
});
const alert = (confidence: number): Alert => ({
  id: "a", asset_id: "PUMP-08", alert_time_utc: "", event_class: "MACHINE", fault_type: "X", confidence, received_at: 0,
});

describe("tile status rules", () => {
  it("red when an alert has confidence >= 0.8", () => expect(tileStatus(asset(), [alert(0.8)], true)).toBe("red"));
  it("amber when alerts are only below 0.8", () => expect(tileStatus(asset(), [alert(0.79)], true)).toBe("amber"));
  it("grey when never seen", () => expect(tileStatus(asset({ messages: 0 }), [], true)).toBe("grey"));
  it("amber when receiving data but not inferred yet", () => expect(tileStatus(asset({ last_infer_utc: null }), [], true)).toBe("amber"));
  it("green when inferred with no alerts", () => expect(tileStatus(asset(), [], true)).toBe("green"));
  it("amber when backend unreachable", () => expect(tileStatus(asset(), [], false)).toBe("amber"));
  it("always lists the 12 default pumps", () => expect(pumpIds({}, []).length).toBe(12));
});
