import type { ReactNode } from "react";

// reference-art-exempt: fixed paper/ink colors intentionally match the supplied static mock
// and are allowed only in this page-local fallback-art component.
const paper = "bg-[#e4d5bd]";
const tilePaper = "bg-[#ddd0ba]";
const ink = "text-[#18120d]";

function ReferenceGlyph({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 100 180" aria-hidden="true" className={className}>
      <path
        fill="currentColor"
        d="M42 0 76 18 79 63 100 86 80 108 78 155 44 180 20 149 20 110 0 86 20 63 20 22Z"
      />
    </svg>
  );
}

export function ReferenceThumb({ children }: { children?: ReactNode }) {
  const hasRealMedia = Boolean(children);
  return (
    <span
      data-reference-art="thumb"
      data-real-media={hasRealMedia ? "true" : "false"}
      className={`relative flex h-11 w-11 shrink-0 items-center justify-center overflow-hidden border border-[color:var(--border-subtle)] ${hasRealMedia ? "bg-transparent" : `rounded-[0.3rem] ${paper}`}`}
    >
      {!hasRealMedia ? (
        <>
          <span className="sr-only">Aperçu mock de glyphe</span>
          <span className="absolute inset-[0.45rem] text-[#1a1410]" aria-hidden="true">
            <ReferenceGlyph className="h-full w-full" />
          </span>
        </>
      ) : null}
      <span className="relative z-10 h-full w-full [&>img]:h-full [&>img]:w-full [&>img]:object-contain">
        {children}
      </span>
    </span>
  );
}

export function ReferenceTileArt({ children }: { children?: ReactNode }) {
  const hasRealMedia = Boolean(children);
  return (
    <span
      data-reference-art="tile"
      data-real-media={hasRealMedia ? "true" : "false"}
      className={`absolute left-3 right-3 top-3 bottom-8 grid place-items-center overflow-hidden ${hasRealMedia ? "bg-transparent" : `rounded-[0.3rem] ${tilePaper}`}`}
    >
      {!hasRealMedia ? (
        <span className={`absolute inset-[24%_35%] ${ink}`} aria-hidden="true">
          <ReferenceGlyph className="h-full w-full" />
        </span>
      ) : null}
      <span className="relative z-10 h-full w-full [&>img]:h-full [&>img]:w-full [&>img]:object-contain">
        {children}
      </span>
    </span>
  );
}
