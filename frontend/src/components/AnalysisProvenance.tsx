import type { SegmentResult } from '../types';

export function AnalysisProvenance({ result }: { result: SegmentResult }) {
  const provenance = result.provenance;
  if (!provenance) return null;
  return (
    <div className="shrink-0 border-b border-[color:var(--border-subtle)] px-3 py-2 text-xs text-[color:var(--text-muted)]" aria-label="Provenance de cette analyse">
      {provenance.classifier_models.map((model) => (
        <div key={model.artifact_id} title={`Empreinte du modèle : ${model.artifact_id}`}>
          Modèle de cette analyse : {model.version_id ?? model.model_version ?? 'version inconnue'} · {model.artifact_id.slice(0, 12)}
          {model.backbone && <span> · {model.backbone.backbone} · {model.backbone.state_sha256?.slice(0, 12)}</span>}
        </div>
      ))}
      {provenance.segmenter && (
        <div title={`Empreinte de segmentation : ${provenance.segmenter.sha256}`}>
          Segmentation : {provenance.segmenter.name} · {provenance.segmenter.sha256.slice(0, 12)}
        </div>
      )}
      {provenance.classifier_models.length === 0 && provenance.correction_count > 0 && (
        <div>Reconnaissance : corrections approuvées, sans prédiction du classifieur.</div>
      )}
      {provenance.correction_count > 0 && (
        <p>{provenance.correction_count} correction(s) approuvée(s) réappliquée(s). Les autres régions restent des propositions du modèle.</p>
      )}
    </div>
  );
}
