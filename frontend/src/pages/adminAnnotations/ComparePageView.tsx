import { useState } from "react";
import { ImageBBoxOverlay, type ImageBBoxStageBox } from "../../components/ImageBBoxStage";
import { adminAnnotationMediaUrl } from "../../services/api";
import type { AdminModelComparison, ClassifyResult } from "../../types";
import { AdminMediaImage } from "./AdminMediaImage";

type Row = AdminModelComparison["rows"][number];
type Model = "base" | "candidate";

const outcomeLabel: Record<Row["outcome"], string> = {
  gain: "Correction", regression: "Régression", both_wrong: "Deux erreurs",
  unchanged: "Sans changement", new_class: "Nouvelle classe", disagreement: "Désaccord",
};
const reviewLabel = { approved: "Validée", pending: "En attente", rejected: "Rejetée" } as const;
function outcome(row: Row) {
  return row.outcome === "gain" && row.base.rejected ? "Abstention levée" : outcomeLabel[row.outcome];
}

function prediction(result: ClassifyResult) {
  return result.rejected ? `Abstention · ${result.class_name} (${Math.round(result.confidence * 100)} %)`
    : `${result.class_name} · ${Math.round(result.confidence * 100)} %`;
}

function ModelPage({
  title, model, source, rows, highlightedIds, selectedId, onSelect,
}: {
  title: string; model: Model; source: Row; rows: Row[]; highlightedIds: ReadonlySet<string>; selectedId: string | null; onSelect: (id: string) => void;
}) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const url = adminAnnotationMediaUrl(source.source_image_url);
  const size = source.source_image_size;
  const boxes: ImageBBoxStageBox[] = rows.map((row) => ({
    id: row.sample_id,
    bbox: row.bbox as [number, number, number, number],
    label: `#${row.index} ${row[model].class_name}`,
    rejected: row[model].rejected,
  }));
  const disagreements = new Set(rows.filter((row) =>
    row.base.class_name !== row.candidate.class_name || row.base.rejected !== row.candidate.rejected
  ).map((row) => row.sample_id));
  const indexes = new Map(rows.map((row) => [row.sample_id, row.index]));

  return (
    <section className="min-w-0 border border-[color:var(--border-subtle)] bg-[color:var(--surface-elevated)] p-3" aria-label={title}>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h4 className="font-semibold text-[color:var(--text-heading)]">{title}</h4>
        <span className="ui-text-caption">{rows.length} régions</span>
      </div>
      {!url || failedUrl === url ? (
        <div role="img" aria-label={`Page source ${source.image_name || source.analysis_id || ""}`} className="ui-empty-state min-h-48">Page source indisponible</div>
      ) : size?.[0] && size?.[1] ? (
        <div className="bg-[color:var(--surface-muted)]">
          <div className="relative mx-auto w-full" style={{ aspectRatio: `${size[0]} / ${size[1]}`, maxWidth: `${38 * size[0] / size[1]}vh` }}>
            <img src={url} alt={`Page ${source.image_name || source.analysis_id}, prédictions du ${title.toLowerCase()}`}
              loading="eager" decoding="async" className="absolute inset-0 h-full w-full object-fill"
              onError={() => setFailedUrl(url)} />
            <ImageBBoxOverlay
              imageSize={size} boxes={boxes} selectedId={selectedId}
              showLabelNames onSelectBox={(id) => { if (id !== null) onSelect(String(id)); }}
              renderLabel={(box, state) => state.focused ? box.label : `#${indexes.get(String(box.id))}`}
              renderBoxExtras={(box, state, bbox) => {
                const muted = highlightedIds.size < rows.length && !highlightedIds.has(String(box.id));
                return muted ? (
                  <rect x={bbox[0]} y={bbox[1]} width={bbox[2]} height={bbox[3]}
                    fill="var(--surface-muted)" fillOpacity={0.6} stroke="var(--border-subtle)"
                    strokeWidth={2} vectorEffect="non-scaling-stroke" />
                ) : disagreements.has(String(box.id)) ? (
                  <rect x={bbox[0]} y={bbox[1]} width={bbox[2]} height={bbox[3]}
                    fill="none" stroke="#ea580c" strokeWidth={state.focused ? 4 : 2}
                    strokeDasharray="6 3" vectorEffect="non-scaling-stroke" />
                ) : (
                  <rect x={bbox[0]} y={bbox[1]} width={bbox[2]} height={bbox[3]}
                    fill="none" stroke="#2563eb" strokeWidth={state.focused ? 3 : 2}
                    vectorEffect="non-scaling-stroke" />
                );
              }}
              svgProps={{ role: "img", "aria-label": `Toutes les bbox du ${title.toLowerCase()}` }}
            />
          </div>
        </div>
      ) : (
        <div>
          <AdminMediaImage src={url} alt={`Page ${source.image_name || source.analysis_id}`} className="max-h-[42rem] w-full object-contain" />
          <p className="ui-text-caption">Dimensions manquantes : les bbox ne peuvent pas être superposées précisément.</p>
        </div>
      )}
      {url ? <a className="ui-action-ghost mt-2 inline-block text-sm" href={url} target="_blank" rel="noopener noreferrer">Ouvrir l’image entière</a> : null}
    </section>
  );
}

function PredictionDetail({ title, result, expected }: { title: string; result: ClassifyResult; expected: string | null }) {
  const verdict = !expected ? "Sans vérité validée" : result.rejected ? "Abstention"
    : result.class_name === expected ? "Correct" : "Erreur";
  return (
    <div className="min-w-0 border border-[color:var(--border-subtle)] p-3">
      <h5 className="font-semibold">{title}</h5>
      <p className="mt-1">{prediction(result)} · {verdict}</p>
      <p className="ui-text-caption mt-2">Top {result.top_k.length}</p>
      <ol className="mt-1 space-y-1 text-sm">
        {result.top_k.map((item, index) =>
          <li key={`${item.class_name}-${index}`}>{index + 1}. {item.class_name} · {Math.round(item.confidence * 100)} %</li>)}
      </ol>
    </div>
  );
}

export function ComparePageView({ rows, visibleRows }: { rows: Row[]; visibleRows: Row[] }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = visibleRows.find((row) => row.sample_id === selectedId) ?? visibleRows[0] ?? null;
  const first = rows[0];
  const disagreements = rows.filter((row) =>
    row.base.class_name !== row.candidate.class_name || row.base.rejected !== row.candidate.rejected).length;
  const labeledRows = rows.filter((row) => row.expected_class);
  const labeled = labeledRows.length;
  const correct = (model: Model) => labeledRows.filter((row) =>
    !row[model].rejected && row[model].class_name === row.expected_class).length;
  const exposure = first?.candidate_exposure;
  const groups: Row["outcome"][] = ["regression", "gain", "new_class", "both_wrong", "disagreement", "unchanged"];
  const highlightedIds = new Set(visibleRows.map((row) => row.sample_id));

  if (!first) return null;
  return (
    <section className="mt-2" aria-label="Analyse visuelle de la page">
      <div className="mb-2">
        <h3 className="ui-title-sm break-all">{first.image_name || first.analysis_id || "Page sans nom"} · {rows.length} région{rows.length > 1 ? "s" : ""} · {disagreements} désaccord{disagreements > 1 ? "s" : ""}</h3>
        <p className="ui-text-caption">Mêmes bbox enregistrées, pas de nouvelle détection · {labeled ? `classe validée : actif ${correct("base")}/${labeled}, candidat ${correct("candidate")}/${labeled} justes` : "aucune classe validée pour mesurer la justesse"} · orange : désaccords, bleu : mêmes prédictions</p>
        {rows.length === 1 ? <p className="ui-text-caption mt-1">Cette page n’a qu’une région enregistrée : une prédiction par modèle est donc affichée.</p> : null}
      </div>
      {first.scope !== "ad_hoc" ? <p role="note" className="ui-alert ui-alert--accent mb-3 p-2 text-sm">Vue partielle : seules les régions approuvées capturées dans le jeu du candidat sont visibles sur cette page.</p> : null}
      {exposure === "unseen_exact" ? <p role="note" className="ui-alert ui-alert--accent mb-3 p-2 text-sm">Page absente du snapshot du candidat par empreinte exacte ; des variantes proches restent possibles.</p> : null}
      {first.scope === "ad_hoc" && exposure !== "train" && exposure !== "unseen_exact" ? <p role="note" className="ui-alert ui-alert--accent mb-3 p-2 text-sm">Comparaison exploratoire : cette page ne constitue pas un test indépendant.</p> : null}
      <div className="grid items-start gap-3 lg:grid-cols-2">
        <ModelPage title="Modèle actif" model="base" source={first} rows={rows} highlightedIds={highlightedIds} selectedId={selected?.sample_id ?? null} onSelect={setSelectedId} />
        <ModelPage title="Modèle candidat" model="candidate" source={first} rows={rows} highlightedIds={highlightedIds} selectedId={selected?.sample_id ?? null} onSelect={setSelectedId} />
      </div>
      {rows.length > 1 ? (
        <nav className="mt-3 border border-[color:var(--border-subtle)] p-3" aria-label="Régions comparées">
          <h4 className="font-semibold">Explorer les régions · {visibleRows.length}/{rows.length} mises en évidence</h4>
          {visibleRows.length ? groups.map((group) => {
            const members = visibleRows.filter((row) => row.outcome === group);
            if (!members.length) return null;
            return (
              <section key={group} className="mt-3" aria-label={outcomeLabel[group]}>
                <h5 className="ui-text-caption font-semibold">{outcomeLabel[group]} · {members.length}</h5>
                <div className="mt-1 flex flex-wrap gap-2">
                  {members.map((row) => (
                    <button key={row.sample_id} type="button" className={`ui-action-ghost border px-3 py-2 text-left text-sm ${selected?.sample_id === row.sample_id ? "border-[color:var(--accent)] bg-[color:var(--surface-muted)]" : "border-[color:var(--border-subtle)]"}`}
                      aria-label={`Région #${row.index} : ${outcome(row)} ; actif ${row.base.class_name} ; candidat ${row.candidate.class_name}`}
                      aria-pressed={selected?.sample_id === row.sample_id} onClick={() => setSelectedId(row.sample_id)}>
                      <strong>#{row.index}</strong> · {row.base.class_name} → {row.candidate.class_name}
                    </button>
                  ))}
                </div>
              </section>
            );
          }) : <p className="ui-text-caption mt-2">Aucune région ne correspond à cette mise en évidence. Les cadres restent visibles sur les deux images.</p>}
        </nav>
      ) : null}
      {selected ? (
        <section className="mt-3 border border-[color:var(--border-subtle)] p-3" aria-label="Détail de la région sélectionnée">
          <h4 className="font-semibold">Région #{selected.index} · {outcome(selected)}</h4>
          <p className="ui-text-caption">Classe validée : {selected.expected_class || "aucune"} · Relecture : {selected.review_status ? reviewLabel[selected.review_status] : selected.scope === "ad_hoc" ? "Non renseignée" : "Validée"} · bbox [x, y, l, h] : [{selected.bbox.join(", ")}]</p>
          <div className="mt-3 grid gap-3 sm:grid-cols-[8rem_minmax(0,1fr)_minmax(0,1fr)]">
            <AdminMediaImage src={adminAnnotationMediaUrl(selected.crop_url)} alt={`Découpe de la région ${selected.index}`} className="h-32 w-full object-contain" />
            <PredictionDetail title="Modèle actif" result={selected.base} expected={selected.expected_class} />
            <PredictionDetail title="Candidat" result={selected.candidate} expected={selected.expected_class} />
          </div>
          <p className="ui-text-caption mt-2">Les confiances sont propres à chaque modèle et ne sont pas directement comparables.</p>
        </section>
      ) : <p className="ui-empty-state mt-3">Aucune région ne correspond au filtre.</p>}
    </section>
  );
}
