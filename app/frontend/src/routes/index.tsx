import { createFileRoute } from "@tanstack/react-router";
import { Console } from "@/components/quaviz/Console";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "QuaViz Fleet Console — Pump fault monitoring" },
      { name: "description", content: "Live operator console for Retrofit AI / QuaViz pump-fault alerts, confidence and recommended checks." },
      { property: "og:title", content: "QuaViz Fleet Console" },
      { property: "og:description", content: "Fleet health at a glance: live pump fault alerts, confidence and what to check next." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Console,
});
