import type { TileStatus } from "@/lib/quaviz/types";

export const tone: Record<TileStatus, { bar: string; soft: string; text: string; border: string; label: string }> = {
  red: { bar: "bg-alarm", soft: "bg-alarm-soft", text: "text-alarm", border: "border-alarm", label: "FAULT" },
  amber: { bar: "bg-warn", soft: "bg-warn-soft", text: "text-warn", border: "border-warn", label: "WATCH" },
  green: { bar: "bg-ok", soft: "bg-ok-soft", text: "text-ok", border: "border-ok/40", label: "HEALTHY" },
  grey: { bar: "bg-idle", soft: "bg-idle-soft", text: "text-muted-foreground", border: "border-border", label: "NO DATA" },
};
