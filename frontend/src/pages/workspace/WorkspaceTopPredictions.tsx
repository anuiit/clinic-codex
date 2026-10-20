import type { ArchetypeAssetsState } from "./useArchetypeAssets";
import { adminAnnotationMediaUrl } from "../../services/api";

type TopPredictionsProps = {
  topKItems: { class_name: string; confidence: number }[];
  archetypes: ArchetypeAssetsState;
  isRejected: boolean;
  labels: { topPredictions: string; archetypeCoverage: string };
};

export default function WorkspaceTopPredictions({ topKItems, archetypes, isRejected, labels }: TopPredictionsProps) {
  return (
    <div className="workspace-inspector-section flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h4 className="ui-title-md">{labels.topPredictions}</h4>
      </div>
      {archetypes.status === "ready" && (
        <div className="ui-text-meta" data-testid="archetype-coverage">
          {labels.archetypeCoverage
            .replace("{covered}", String(archetypes.covered))
            .replace("{total}", String(archetypes.total))}
        </div>
      )}
      <div className="space-y-2">
        {topKItems.map((item, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="ui-text-meta w-4 text-right">{i + 1}</span>
            {archetypes.status === "ready" && (
              <span
                className="flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden border border-[color:var(--border-subtle)]"
                data-archetype-thumb={item.class_name}
                data-has-asset={archetypes.assets.get(item.class_name) ? "true" : "false"}
              >
                {archetypes.assets.get(item.class_name) ? (
                  <img
                    src={adminAnnotationMediaUrl(archetypes.assets.get(item.class_name) as string)}
                    alt=""
                    className="h-full w-full object-contain"
                    loading="lazy"
                  />
                ) : (
                  <span className="ui-text-meta" aria-hidden="true">
                    —
                  </span>
                )}
              </span>
            )}
            <div className="flex-1">
              <div className="mb-0.5 flex justify-between text-sm">
                <span className={i === 0 ? "font-medium text-[var(--text-body)]" : "text-[color:var(--text-muted)]"}>
                  {item.class_name}
                </span>
                <span className={i === 0 ? "font-medium text-[var(--text-soft)]" : "text-[color:var(--text-muted)]"}>
                  {(item.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="ui-progress-track h-1 w-full">
                <div
                  className={`h-full rounded-full ${i === 0 ? (isRejected ? "ui-progress-value--accent" : "ui-progress-value--ready") : "bg-[var(--surface-hover)]"}`}
                  style={{ width: `${item.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
