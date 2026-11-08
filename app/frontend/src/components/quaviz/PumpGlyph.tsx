import type { TileStatus } from "@/lib/quaviz/types";

/** Stylized centrifugal pump — impeller motion tracks tile status. */
export function PumpGlyph({ status, className = "" }: { status: TileStatus; className?: string }) {
  const motion =
    status === "green"
      ? "animate-impeller"
      : status === "amber"
        ? "animate-impeller-slow"
        : status === "red"
          ? "animate-pump-shake"
          : "";
  const stroke =
    status === "red"
      ? "var(--alarm)"
      : status === "amber"
        ? "var(--warn)"
        : status === "green"
          ? "var(--ok)"
          : "var(--idle)";
  const fill =
    status === "grey"
      ? "color-mix(in oklab, var(--idle) 18%, transparent)"
      : `color-mix(in oklab, ${stroke} 22%, transparent)`;

  return (
    <svg
      viewBox="0 0 64 64"
      className={`h-14 w-14 shrink-0 ${className}`}
      aria-hidden
    >
      {/* base / skid */}
      <rect x="10" y="50" width="44" height="4" rx="1" fill={stroke} opacity="0.35" />
      {/* casing */}
      <circle cx="28" cy="30" r="16" fill={fill} stroke={stroke} strokeWidth="2.2" />
      <circle cx="28" cy="30" r="11" fill="none" stroke={stroke} strokeWidth="1.2" opacity="0.45" />
      {/* discharge flange */}
      <path
        d="M42 24 h10 a2 2 0 0 1 2 2 v8 a2 2 0 0 1 -2 2 H42"
        fill={fill}
        stroke={stroke}
        strokeWidth="2"
        strokeLinejoin="round"
      />
      {/* suction inlet */}
      <path
        d="M12 30 H4"
        stroke={stroke}
        strokeWidth="3"
        strokeLinecap="round"
        opacity="0.7"
      />
      {/* impeller */}
      <g className={`origin-center ${motion}`} style={{ transformOrigin: "28px 30px" }}>
        <path
          d="M28 30 L28 19 M28 30 L37.5 35.5 M28 30 L18.5 35.5"
          stroke={stroke}
          strokeWidth="2.4"
          strokeLinecap="round"
        />
        <circle cx="28" cy="30" r="3.2" fill={stroke} />
      </g>
      {/* flow dashes on discharge when running */}
      {(status === "green" || status === "amber") && (
        <g className="animate-flow" stroke={stroke} strokeWidth="1.5" strokeLinecap="round" opacity="0.7">
          <path d="M46 28 h6" />
          <path d="M48 32 h5" />
        </g>
      )}
      {/* fault spark */}
      {status === "red" && (
        <path
          d="M48 16 l2 4 h4 l-3.2 2.4 1.2 4 -4 -2.4 -4 2.4 1.2 -4 -3.2 -2.4 h4 z"
          fill={stroke}
          className="animate-fault-blink"
        />
      )}
    </svg>
  );
}
