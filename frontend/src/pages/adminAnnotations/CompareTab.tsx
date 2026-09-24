import axios from "axios";
import { useEffect, useMemo, useState } from "react";
import { ActionButton } from "../../components/ui/AdminPrimitives";
import { getAdminTrainingJob, getComparableModels, getLatestAdminTrainingJob, startModelComparison } from "../../services/api";
import type { AdminAnnotationQueue, AdminModelComparison, AdminTrainingJob } from "../../types";
import { TRAINING_JOB_POLL_INTERVAL_MS } from "./model";
import { ComparePageView } from "./ComparePageView";

function ratio(correct: number, support: number) {
  return support ? `${correct}/${support} (${Math.round(correct / support * 100)} %)` : "—";
}

const SCOPE = { train: "Page d’apprentissage", locked_test: "Test réservé", ad_hoc: "Page exploratoire" } as const;
export function CompareTab({
  initialVersionId,
  queue,
  canRunTraining = true,
}: {
  initialVersionId?: string;
  queue: AdminAnnotationQueue | null;
  canRunTraining?: boolean;
}) {
  const [versions, setVersions] = useState<Array<{ version_id: string; status: string }>>([]);
  const [versionId, setVersionId] = useState(initialVersionId ?? "");
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [job, setJob] = useState<AdminTrainingJob | null>(null);
  const [filter, setFilter] = useState("all");
  const [pageKey, setPageKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    let active = true;
    void Promise.all([getComparableModels(), getLatestAdminTrainingJob().catch(() => ({ job: null }))]).then(([result, previous]) => {
      if (!active) return;
      const candidates = result.versions.filter((item) => item.status === "candidate");
      const restored = previous.job?.kind === "comparison" &&
        (!initialVersionId || previous.job.model_version_id === initialVersionId) &&
        candidates.some((item) => item.version_id === previous.job?.model_version_id) ? previous.job : null;
      setVersions(candidates);
      setVersionId((current) => initialVersionId && candidates.some((item) => item.version_id === initialVersionId)
        ? initialVersionId : restored?.model_version_id
          ? restored.model_version_id : current && candidates.some((item) => item.version_id === current)
          ? current : candidates[0]?.version_id ?? "");
      setJob((current) => initialVersionId && current?.model_version_id !== initialVersionId ? restored : current ?? restored);
      setFilter("all");
      setPageKey("");
    }).catch(() => { if (active) setError("Versions candidates indisponibles."); });
    return () => { active = false; };
  }, [initialVersionId]);

  useEffect(() => {
    if (job?.status !== "running") return;
    const runId = job.run_id;
    const interval = window.setInterval(() => {
      void getAdminTrainingJob(runId)
        .then((response) => { if (response.job?.run_id === runId) setJob(response.job); })
        .catch(() => setError("Impossible d’actualiser la comparaison."));
    }, TRAINING_JOB_POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [job?.run_id, job?.status]);

  const report = job?.status === "succeeded" ? job.comparison : null;
  const pages = useMemo(() => {
    const grouped = new Map<string, AdminModelComparison["rows"]>();
    for (const row of report?.rows ?? []) {
      const key = row.source_image_sha256 || `${row.analysis_id}:${row.image_name || row.source_image_url}`;
      grouped.set(key, [...(grouped.get(key) ?? []), row]);
    }
    return [...grouped].map(([key, rows]) => ({ key, rows }));
  }, [report]);
  const page = pages.find((item) => item.key === pageKey)
    ?? pages.find((item) => item.rows[0]?.scope === "locked_test") ?? pages[0];
  const rows = useMemo(() => (page?.rows ?? []).filter((row) => {
    const matchingOutcome = filter === "all" || row.outcome === filter ||
      (filter === "disagreement" && (row.base.class_name !== row.candidate.class_name || row.base.rejected !== row.candidate.rejected));
    return matchingOutcome;
  }), [page, filter]);
  const analyses = [...(queue?.analyses ?? [])]
    .filter((analysis) => analysis.image_exists && analysis.elements.length)
    .sort((left, right) => right.elements.length - left.elements.length);
  const effectiveAnalysisId = analysisId ?? analyses[0]?.analysis_id ?? "";
  const selectedAnalysis = analyses.find((analysis) => analysis.analysis_id === effectiveAnalysisId);

  const clearReport = () => {
    setJob(null);
    setFilter("all");
    setPageKey("");
    setError(null);
  };
  const start = async () => {
    if (!versionId || !canRunTraining) return;
    setStarting(true);
    clearReport();
    try {
      const response = await startModelComparison(versionId, effectiveAnalysisId || undefined);
      setJob(response.job);
    } catch (issue) {
      const backendMessage = axios.isAxiosError(issue) ? String(issue.response?.data?.error ?? "") : "";
      setError(backendMessage || "Comparaison impossible. Essayez avec une page analysée si ce candidat ancien ne contient pas de snapshot compatible.");
    } finally {
      setStarting(false);
    }
  };

  return (
    <section className="overflow-auto p-4" aria-label="Comparaison des modèles">
      <h2 className="ui-title-md">Comparer les modèles sur une page</h2>
      <div className="mt-2 flex flex-wrap items-end gap-3 border border-[color:var(--border-subtle)] bg-[color:var(--surface-elevated)] p-3">
        <label className="admin-field"><span>Version candidate</span>
          <select className="ui-select px-2 py-1" value={versionId} disabled={starting || job?.status === "running"} onChange={(event) => { setVersionId(event.target.value); clearReport(); }}>
            {versions.map((item) => <option key={item.version_id} value={item.version_id}>{item.version_id}</option>)}
          </select>
        </label>
        <label className="admin-field"><span>Source des images</span>
          <select className="ui-select px-2 py-1" value={effectiveAnalysisId} disabled={starting || job?.status === "running"} onChange={(event) => { setAnalysisId(event.target.value); clearReport(); }}>
            <option value="">Jeu figé du candidat · extraits seulement</option>
            {analyses.map((analysis) => (
              <option key={analysis.analysis_id} value={analysis.analysis_id}>
                Page analysée · {analysis.image_name || analysis.analysis_id} · {analysis.elements.length} région{analysis.elements.length > 1 ? "s" : ""}
              </option>
            ))}
          </select>
        </label>
        <ActionButton tone="primary" className="px-3 py-2" disabled={!canRunTraining || !versionId || starting || job?.status === "running"} onClick={() => void start()}>
          {starting ? "Démarrage…" : job?.status === "running" ? "Comparaison en cours…" : "Comparer"}
        </ActionButton>
      </div>
      {!canRunTraining ? <p className="ui-text-caption mt-2">Lecture seule : autorisation de lancement requise.</p> : null}
      {selectedAnalysis?.elements.length === 1 ? <p className="ui-text-caption mt-2">
        Une seule bbox enregistrée sur cette page. Pour comparer plusieurs zones sur une même image, annotez et enregistrez d’abord les autres régions de cette page.
      </p> : null}
      {!versions.length ? <p className="mt-3">Aucun candidat disponible.</p> : null}
      {error ? <div role="alert" className="ui-alert ui-alert--danger mt-3 p-3">{error}</div> : null}
      {job?.status === "running" ? <p role="status" className="mt-3">{job.stage || "Comparaison en cours…"}</p> : null}
      {job?.status === "failed" ? <div role="alert" className="ui-alert ui-alert--danger mt-3 p-3">{job.error || "La comparaison a échoué."}</div> : null}

      {report ? (
        <>
          <section className="ui-alert ui-alert--accent mt-3 p-3" aria-label="Résumé de la comparaison">
            <h3 className="ui-title-sm">
              Bilan du groupe · {SCOPE[report.protocol.metrics_scope as keyof typeof SCOPE] || "Comparaison"} · {report.sample_count} région{report.sample_count > 1 ? "s" : ""} capturée{report.sample_count > 1 ? "s" : ""} au total · {report.metrics.common.gains} correction{report.metrics.common.gains > 1 ? "s" : ""} · {report.metrics.common.regressions} régression{report.metrics.common.regressions > 1 ? "s" : ""}
            </h3>
            <p className="ui-text-caption mt-1">
              {report.protocol.metrics_scope === "locked_test"
                ? "Résultats descriptifs sur le test réservé ; l’indépendance historique du modèle actif n’est pas établie."
                : "Ces résultats décrivent le comportement sur ces découpes ; ils ne mesurent pas la généralisation."}
              {page?.rows[0]?.candidate_exposure === "train" ? " Cette page a servi à entraîner le candidat." : ""}
              {report.metrics.new_classes.support > 0 ? ` ${report.metrics.new_classes.support} nouvelle(s) classe(s), non comparables au modèle actif.` : ""}
            </p>
          </section>
          <div className="mt-2 flex flex-wrap items-end gap-3">
            {pages.length > 1 ? <label className="admin-field"><span>Page à inspecter</span>
              <select className="ui-select max-w-[min(32rem,85vw)] px-2 py-1" value={page?.key ?? ""} onChange={(event) => setPageKey(event.target.value)}>
                {pages.map((item) => <option key={item.key} value={item.key}>
                  {item.rows[0].image_name || item.rows[0].analysis_id || item.key.slice(0, 12)} · {item.rows.length} régions · {SCOPE[item.rows[0].scope]}
                </option>)}
              </select>
            </label> : null}
            <label className="admin-field"><span>Mettre en évidence</span>
              <select className="ui-select px-2 py-1" value={filter} onChange={(event) => setFilter(event.target.value)}>
                <option value="all">Toutes les régions</option><option value="disagreement">Désaccords</option>
                <option value="gain">Corrections</option><option value="regression">Régressions</option>
                <option value="both_wrong">Deux erreurs</option><option value="new_class">Nouvelles classes</option>
              </select>
            </label>
          </div>
          {page && filter !== "all" ? <p className="ui-text-caption mt-2">Mise en évidence : {rows.length}/{page.rows.length} régions. Les autres cadres restent visibles.</p> : null}
          {page && page.rows[0].scope !== report.protocol.metrics_scope ? (
            <p role="note" className="ui-text-caption mt-2">Page {SCOPE[page.rows[0].scope]} : les chiffres ci-dessus restent ceux du groupe {SCOPE[report.protocol.metrics_scope as keyof typeof SCOPE] || report.protocol.metrics_scope} ; ils ne sont pas recalculés pour cette page.</p>
          ) : null}
          {page ? <ComparePageView key={page.key} rows={page.rows} visibleRows={rows} /> : <p className="ui-empty-state mt-3">Aucune région capturée pour cette comparaison.</p>}
          <details className="mt-5 border border-[color:var(--border-subtle)] p-3">
            <summary className="cursor-pointer font-semibold">Mesures et protocole</summary>
            <p className="ui-text-caption mt-2">Groupe mesuré : {SCOPE[report.protocol.metrics_scope as keyof typeof SCOPE] || report.protocol.metrics_scope}. Les chiffres du groupe apprentissage ne sont pas un test de généralisation.</p>
            <p className="ui-text-caption mt-2">Découpes enregistrées calculées sur {report.protocol.device} · indépendance vis-à-vis de l’apprentissage historique du modèle actif : {report.protocol.base_historical_independence === "unknown" ? "inconnue" : report.protocol.base_historical_independence}</p>
            <ComparisonMetrics report={report} />
            {report.warnings.map((warning) => <p key={warning} className="ui-text-caption mt-2">{warning}</p>)}
          </details>
        </>
      ) : null}
    </section>
  );
}

function ComparisonMetrics({ report }: { report: AdminModelComparison }) {
  const { common, new_classes: newClasses, coverage, top3 } = report.metrics;
  return (
    <dl className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <div><dt>Classes communes · exacts</dt><dd>Actif {ratio(common.base_correct, common.support)} · candidat {ratio(common.candidate_correct, common.support)}</dd></div>
      <div><dt>Gains / régressions</dt><dd>{common.gains} / {common.regressions} · {common.both_wrong} erreurs communes</dd></div>
      <div><dt>Nouvelles classes</dt><dd>Candidat {ratio(newClasses.candidate_correct, newClasses.support)}</dd></div>
      <div><dt>Top 3 classes communes</dt><dd>Actif {ratio(top3.base_correct, common.support)} · candidat {ratio(top3.candidate_correct, common.support)}</dd></div>
      <div><dt>Couverture · prédictions acceptées, pas justesse</dt><dd>Actif {Math.round(coverage.base * 100)} % · candidat {Math.round(coverage.candidate * 100)} %</dd></div>
    </dl>
  );
}
