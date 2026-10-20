import axios from "axios";
import { useCallback, useEffect, useState } from "react";
import { ActionButton, AdminSection } from "../../components/ui/AdminPrimitives";
import { getAdminTrainingJob, getAdminTrainingSummary, startAdminTrainingJob } from "../../services/api";
import type { AdminTrainingJob, AdminTrainingSummary } from "../../types";
import "./TrainingTab.module.css";
import { TRAINING_JOB_POLL_INTERVAL_MS } from "./model";
import { PanelSkeleton } from "./shared";

function launchReason(reason: string) {
  if (reason.startsWith("legacy_reviews_require_import")) return "Les anciennes validations doivent être importées par l’administrateur avant l’entraînement. Sauvegardez les annotations, lancez l’import local, puis redémarrez l’application.";
  if (reason.includes("no_trainable_annotations")) return "Validez au moins une annotation utilisable dans Trier.";
  if (reason.includes("unknown") || reason.includes("unconfirmed")) return "Confirmez les nouvelles classes dans Classes.";
  if (reason.includes("backbone")) return "Le modèle DINOv2 requis est absent ; vérifiez l’installation locale.";
  if (reason.includes("running")) return "Une tâche est déjà en cours.";
  return reason.replaceAll("_", " ");
}

function elapsed(startedAt?: string) {
  if (!startedAt) return "";
  const seconds = Math.max(0, Math.floor((Date.now() - Date.parse(startedAt)) / 1000));
  return Number.isFinite(seconds) ? `${Math.floor(seconds / 60)} min ${seconds % 60} s` : "";
}

export function TrainingTab({ onCompareCandidate, canRunTraining = true }: { onCompareCandidate?: (versionId: string) => void; canRunTraining?: boolean }) {
  const [summary, setSummary] = useState<AdminTrainingSummary | null>(null);
  const [job, setJob] = useState<AdminTrainingJob | null>(null);
  const [verified, setVerified] = useState<AdminTrainingJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [device, setDevice] = useState("auto");
  const [batchSize, setBatchSize] = useState(16);
  const [notes, setNotes] = useState("");
  const [, tick] = useState(0);

  const refresh = useCallback(async () => {
    const next = await getAdminTrainingSummary();
    setSummary(next);
    const latest = next.latest_training_job ?? (next.latest_job?.kind === "comparison" ? null : next.latest_job ?? null);
    setJob(latest);
    if (latest?.dry_run && latest.status === "succeeded") setVerified(latest);
    return next;
  }, []);

  useEffect(() => {
    let active = true;
    void getAdminTrainingSummary().then((next) => {
      if (!active) return;
      setSummary(next);
      const latest = next.latest_training_job ?? (next.latest_job?.kind === "comparison" ? null : next.latest_job ?? null);
      setJob(latest);
      if (latest?.dry_run && latest.status === "succeeded") setVerified(latest);
      setDevice(next.parameters.editable.device[0] ?? "auto");
      setBatchSize(next.parameters.editable.batch_size.default);
    }).catch(() => { if (active) setError("Résumé d’entraînement indisponible."); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (job?.status !== "running") return;
    const runId = job.run_id;
    const interval = window.setInterval(() => {
      tick((value) => value + 1);
      void getAdminTrainingJob(runId).then(({ job: next }) => {
        if (!next || next.run_id !== runId) return;
        setJob(next);
        if (next.dry_run && next.status === "succeeded") setVerified(next);
      }).catch(() => setError("Impossible d’actualiser la tâche."));
    }, TRAINING_JOB_POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [job?.run_id, job?.status]);

  if (loading) return <PanelSkeleton label="Chargement de l'entraînement" />;
  if (!summary) return <div role="alert" className="ui-alert ui-alert--danger p-4">{error}</div>;

  const revision = summary.training_snapshot.data_revision;
  const readyFromDryRun = Boolean(
    verified?.kind !== "comparison" &&
    verified?.dry_run &&
    verified.status === "succeeded" &&
    revision &&
    verified.training_snapshot?.data_revision === revision &&
    verified.device === device &&
    verified.batch_size === batchSize,
  );
  const candidateId = job?.status === "succeeded" && !job.dry_run ? job.model_version_id : null;
  const result = candidateId ? job?.result : null;
  const running = job?.status === "running";
  const blocked = !summary.launch_allowed_for_request;
  const bounds = summary.parameters.editable.batch_size;
  const actionLabel = candidateId ? "Comparer les modèles" : readyFromDryRun ? "Créer un candidat" : "Vérifier la préparation";

  const start = async () => {
    if (candidateId) {
      onCompareCandidate?.(candidateId);
      return;
    }
    if (!canRunTraining) return;
    if (!Number.isInteger(batchSize) || batchSize < bounds.min || batchSize > bounds.max) {
      setError(`Taille de lot : entier entre ${bounds.min} et ${bounds.max}.`);
      return;
    }
    if (notes.length > 200) {
      setError("Notes : 200 caractères maximum.");
      return;
    }
    setError(null);
    setStarting(true);
    try {
      const current = await getAdminTrainingSummary();
      setSummary(current);
      if (current.training_snapshot.data_revision !== revision || !current.launch_allowed_for_request) {
        setVerified(null);
        setError("Les données ou les prérequis ont changé. Vérifiez la préparation à nouveau.");
        return;
      }
      const response = await startAdminTrainingJob({
        dry_run: !readyFromDryRun,
        device,
        batch_size: batchSize,
        notes,
        expected_data_revision: revision,
      });
      setJob(response.job);
    } catch (issue) {
      if (axios.isAxiosError(issue) && issue.response?.status === 409) {
        setVerified(null);
        setError("Les données ont changé pendant le lancement. Vérifiez à nouveau.");
        void refresh().catch(() => undefined);
      } else {
        const reason = axios.isAxiosError(issue) ? issue.response?.data?.error : null;
        setError(typeof reason === "string" && reason ? launchReason(reason) : "Lancement impossible. Vérifiez le détail technique ou réessayez.");
      }
    } finally {
      setStarting(false);
    }
  };

  return (
    <section className="overflow-auto p-4" aria-label="Entraînement">
      <AdminSection as="div" variant="summary">
        <h2 className="ui-title-md">Créer un candidat</h2>
        <p className="ui-text-caption">Le candidat utilisera le modèle existant et les annotations validées. Il ne sera pas activé automatiquement.</p>
        <p className="mt-2 text-sm">{summary.data.trainable} annotations utilisables · {summary.data.classes.length} classes représentées</p>
        {summary.training_snapshot.new_classes?.length ? (
          <p className="mt-1 text-sm">Nouvelles classes incluses : {summary.training_snapshot.new_classes.join(", ")}</p>
        ) : null}
      </AdminSection>

      {blocked ? (
        <div role="alert" className="ui-alert ui-alert--accent mt-3 p-3">
          <strong>Préparation bloquée</strong>
          <ul className="mt-1 list-disc pl-5">{summary.launch_disabled_reasons.map((reason) => <li key={reason}>{launchReason(reason)}</li>)}</ul>
        </div>
      ) : null}
      {!canRunTraining ? <p className="ui-text-caption mt-2">Lecture seule : autorisation de lancement requise.</p> : null}
      {error ? <div role="alert" className="ui-alert ui-alert--danger mt-3 p-3">{error}</div> : null}

      <AdminSection className="mt-3">
        <h3 className="ui-title-sm">
          {running ? "Tâche en cours" : candidateId ? "Candidat créé" : readyFromDryRun ? "Préparation vérifiée" : "Prêt à vérifier"}
        </h3>
        {running ? (
          <p role="status" className="mt-2">
            {job?.stage || (job?.dry_run ? "Vérification" : "Entraînement")} · {elapsed(job?.started_at)}
          </p>
        ) : null}
        {job?.status === "failed" ? <p role="alert" className="mt-2 text-[color:var(--danger-text)]">{job.error || "La tâche a échoué."}</p> : null}
        {job?.status === "succeeded" && job.dry_run ? (
          <p className="mt-2">Essai à blanc réussi : aucun candidat n’a été créé.</p>
        ) : null}
        {candidateId ? <p className="mt-2">Version {candidateId} prête à comparer. Le modèle utilisé par l’application est inchangé.</p> : null}
        {result ? (
          <div className="mt-3 border-t border-[color:var(--border-subtle)] pt-3" aria-label="Résultats du candidat">
            {result.train_count ? <p>Apprentissage : actuel {result.active_correct}/{result.train_count} · candidat {result.candidate_correct}/{result.train_count}.</p> : null}
            {result.holdout?.support ? <p className="mt-1">Test réservé : actuel {result.holdout.base_correct}/{result.holdout.support} · candidat {result.holdout.candidate_correct}/{result.holdout.support}.</p> : null}
            <p className="ui-text-caption mt-1">Généralisation non validée : ces résultats ne prouvent pas une amélioration sur d’autres images.</p>
          </div>
        ) : null}
        <ActionButton tone="primary" className="mt-4 px-4 py-2" disabled={starting || running || (!canRunTraining && !candidateId) || (blocked && !candidateId)} onClick={() => void start()}>
          {starting ? "Démarrage…" : running ? "En cours…" : actionLabel}
        </ActionButton>
        {candidateId ? (
          <ActionButton tone="ghost" className="ml-2 px-2 py-1 text-sm" onClick={() => { setJob(null); setVerified(null); }}>
            Préparer un autre candidat
          </ActionButton>
        ) : null}
      </AdminSection>

      <details className="mt-3 p-2">
        <summary>Options avancées</summary>
        <div className="mt-2 flex flex-wrap gap-3">
          <label className="admin-field"><span>Machine</span><select className="ui-select px-2 py-1" value={device} onChange={(event) => { setDevice(event.target.value); setVerified(null); }}>
            {summary.parameters.editable.device.map((value) => <option key={value} value={value}>{value}</option>)}
          </select></label>
          <label className="admin-field"><span>Taille de lot</span><input className="ui-input px-2 py-1" type="number" min={bounds.min} max={bounds.max} value={batchSize} onChange={(event) => { setBatchSize(Number(event.target.value)); setVerified(null); }} /></label>
          <label className="admin-field"><span>Notes</span><input className="ui-input px-2 py-1" value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={200} /></label>
        </div>
      </details>
      <details className="mt-1 p-2">
        <summary>Détails techniques et journal</summary>
        <p className="mt-2 break-all text-xs">Révision des données : {revision || "indisponible"}</p>
        <p className="text-xs">Exécution : {job?.run_id || "aucune"}</p>
        {job?.log_tail?.length ? <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap text-xs">{job.log_tail.join("\n")}</pre> : null}
      </details>
    </section>
  );
}
