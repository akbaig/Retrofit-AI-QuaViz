import { toast } from "sonner";
import { humanFault, pct, RED_CONFIDENCE } from "./status";
import type { Alert, Explanation } from "./types";

let permissionAsked = false;

/** Ask once for browser notifications (advisory; never blocks the UI). */
export function ensureNotifyPermission() {
  if (typeof window === "undefined" || !("Notification" in window)) return;
  if (permissionAsked || Notification.permission !== "default") return;
  permissionAsked = true;
  void Notification.requestPermission().catch(() => undefined);
}

function browserNotify(title: string, body: string, tag?: string) {
  if (typeof window === "undefined" || !("Notification" in window)) return;
  if (Notification.permission !== "granted") return;
  try {
    const opts: NotificationOptions = { body, silent: false };
    if (tag) opts.tag = tag;
    new Notification(title, opts);
  } catch {
    /* ignore — some browsers require a service worker */
  }
}

export function notifyAlert(a: Alert) {
  const title = `${a.asset_id} · ${humanFault(a.fault_type)}`;
  const body = `${pct(a.confidence)} · ${a.recommended_action?.slice(0, 120) || "Check evidence"}`;
  const red = a.confidence >= RED_CONFIDENCE;
  toast[red ? "error" : "warning"](title, { description: body, duration: 8000 });
  browserNotify(title, body, `alert-${a.id}`);
}

export function notifyExplain(e: Explanation) {
  const summary = e.narrative?.summary;
  if (!summary) return;
  const title = `${e.asset_id} · explanation`;
  toast.message(title, { description: String(summary).slice(0, 160), duration: 6000 });
}
