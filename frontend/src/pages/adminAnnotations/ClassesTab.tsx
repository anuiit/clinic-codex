import axios from "axios";
import { useEffect, useMemo, useState } from "react";
import { adminAnnotationMediaUrl, confirmAdminClass, getAdminClasses } from "../../services/api";
import type { AdminAnnotationQueue, AdminClassCatalogue } from "../../types";
import { AdminMediaImage } from "./AdminMediaImage";

type ClassRow = AdminClassCatalogue["classes"][number];

export function ClassesTab({
  queue,
  onViewAnnotations,
  onConfirmed,
  canConfirm = true,
}: {
  queue?: AdminAnnotationQueue | null;
  onViewAnnotations: (className: string) => void;
  onConfirmed: (names: string[]) => void;
  canConfirm?: boolean;
}) {
  const [catalogue, setCatalogue] = useState<AdminClassCatalogue | null>(null);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let active = true;
    void getAdminClasses()
      .then((result) => { if (active) { setCatalogue(result); setError(null); } })
      .catch(() => { if (active) setError("Catalogue des classes indisponible."); });
    return () => { active = false; };
  }, [reload]);

  const groups = useMemo(() => {
    const rows = (catalogue?.classes ?? []).filter((item) =>
      item.class_name.toLocaleLowerCase("fr").includes(query.trim().toLocaleLowerCase("fr")));
    return {
      unconfirmed: rows.filter((item) => item.status === "unconfirmed"),
      candidate: rows.filter((item) => item.status === "candidate"),
      active: rows.filter((item) => item.status === "active"),
    };
  }, [catalogue, query]);

  const classMedia = useMemo(() => {
    const byClass = new Map<string, { images: string[]; pendingImages: string[]; pending: number }>();
    for (const analysis of queue?.analyses ?? []) {
      for (const element of analysis.elements) {
        const entry = byClass.get(element.class_name) ?? { images: [], pendingImages: [], pending: 0 };
        if (element.crop_exists && entry.images.length < 2) entry.images.push(element.crop_url);
        if (element.review_status === "pending") {
          entry.pending += 1;
          if (element.crop_exists && entry.pendingImages.length < 2) entry.pendingImages.push(element.crop_url);
        }
        byClass.set(element.class_name, entry);
      }
    }
    return byClass;
  }, [queue]);
  const illustrated = (catalogue?.classes ?? []).filter((item) =>
    (classMedia.get(item.class_name)?.pending ?? 0) > 0 &&
    (classMedia.get(item.class_name)?.pendingImages.length ?? 0) > 0 &&
    item.class_name.toLocaleLowerCase("fr").includes(query.trim().toLocaleLowerCase("fr")),
  ).slice(0, 6);

  const confirm = async (className: string) => {
    if (!catalogue || !canConfirm || !window.confirm(
      `Confirmer « ${className} » ? Un identifiant de classe sera attribué définitivement. Vérifiez le nom et les annotations ; le modèle actif ne changera pas.`,
    )) return;
    setBusy(className);
    setError(null);
    try {
      const next = await confirmAdminClass(className, catalogue.revision);
      setCatalogue(next);
      onConfirmed(next.classes.filter((item) => item.status !== "unconfirmed").map((item) => item.class_name));
    } catch (issue) {
      if (axios.isAxiosError(issue) && issue.response?.status === 409) {
        const code = issue.response.data?.error_code;
        setError(code === "REVIEW_MIGRATION_REQUIRED"
          ? "Importez les anciennes validations avant de confirmer une classe."
          : code === "CLASS_NAME_SIMILAR"
            ? "Une classe au nom similaire existe déjà. Vérifiez l’orthographe et utilisez la classe existante."
            : "Le catalogue a changé. Vérifiez les classes actualisées avant de confirmer.");
        void getAdminClasses().then(setCatalogue).catch(() => undefined);
      } else {
        setError("Impossible de confirmer cette classe. Vérifiez les droits et le nom.");
      }
    } finally {
      setBusy(null);
    }
  };

  const classCard = (item: ClassRow) => {
    const count = item.counts.pending + item.counts.approved + item.counts.rejected;
    const images = classMedia.get(item.class_name)?.images ?? [];
    return (
      <li key={item.class_name} className="rounded-lg border border-[color:var(--border-subtle)] bg-[color:var(--surface-muted)] p-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <strong className="text-[color:var(--text-heading)]">{item.class_name}</strong>
            <p className="ui-text-caption">
              {item.class_label === null ? "Pas encore d’identifiant" : `ID ${item.class_label}`}
              {" · "}{item.counts.pending} à vérifier · {item.counts.approved} validées · {item.counts.rejected} rejetées
            </p>
            {item.status === "candidate" ? <p className="ui-text-caption">Prête pour le prochain candidat · {item.trainable_count} utilisable(s)</p> : null}
          </div>
          <div className="flex flex-wrap gap-2">
            {count > 0 ? <button type="button" className="ui-action-ghost px-2 py-1" onClick={() => onViewAnnotations(item.class_name)}>Voir les annotations</button> : null}
            {item.status === "unconfirmed" ? (
              <button type="button" className="ui-action-ghost px-2 py-1" disabled={!canConfirm || busy !== null} onClick={() => void confirm(item.class_name)}>
                {busy === item.class_name ? "Confirmation…" : "Confirmer la classe"}
              </button>
            ) : null}
          </div>
        </div>
        {images.length ? (
          <div className="mt-3 flex gap-2" aria-label={`Images de la classe ${item.class_name}`}>
            {images.map((url, index) => (
              <AdminMediaImage
                key={url}
                src={adminAnnotationMediaUrl(url)}
                alt={`Exemple ${index + 1} de ${item.class_name}`}
                loading="lazy"
                className="h-16 w-16 rounded-md border border-[color:var(--border-subtle)] object-contain"
              />
            ))}
          </div>
        ) : null}
      </li>
    );
  };

  return (
    <section className="overflow-auto p-4" aria-label="Gestion des classes">
      <h2 className="ui-title-md">Classes disponibles</h2>
      <p className="ui-text-caption">Confirmez les nouveaux noms avant de créer un candidat. Le modèle actif reste inchangé.</p>
      <label className="admin-field mt-3 max-w-sm">
        <span>Rechercher une classe</span>
        <input className="ui-input px-2 py-1" type="search" value={query} onChange={(event) => setQuery(event.target.value)} />
      </label>
      {error ? <div role="alert" className="ui-alert ui-alert--danger mt-3 p-3">{error}</div> : null}
      {!catalogue && error ? <button type="button" className="ui-action-ghost mt-2 px-3 py-2" onClick={() => { setError(null); setReload((value) => value + 1); }}>Réessayer</button> : null}
      {!canConfirm ? <p className="ui-text-caption mt-3">Confirmation indisponible : autorisation de validation ou import des anciennes décisions requis.</p> : null}
      {!catalogue ? error ? null : <p className="mt-4">Chargement des classes…</p> : (
        <div className="mt-4 space-y-5">
          {illustrated.length ? (
            <section aria-label="Classes illustrées">
              <h3 className="ui-title-sm">À examiner en images</h3>
              <p className="ui-text-caption">Exemples en attente de validation : ils ne servent pas encore à l’entraînement.</p>
              <ul className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {illustrated.map((item) => (
                  <li key={item.class_name}>
                    <button type="button" className="ui-action-ghost flex w-full items-center gap-3 p-2 text-left" onClick={() => onViewAnnotations(item.class_name)}>
                      <AdminMediaImage
                        src={adminAnnotationMediaUrl(classMedia.get(item.class_name)!.pendingImages[0])}
                        alt={`Exemple de ${item.class_name}`}
                        className="h-14 w-14 shrink-0 rounded-md object-contain"
                      />
                      <span><strong className="block">{item.class_name}</strong><span className="ui-text-caption">{classMedia.get(item.class_name)?.pending} à vérifier</span></span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
          {groups.unconfirmed.length > 0 ? <section aria-label="Classes à confirmer">
            <h3 className="ui-title-sm">À confirmer <span className="ui-text-caption">({groups.unconfirmed.length})</span></h3>
            <p className="ui-text-caption">Vérifiez le nom : la confirmation attribue un ID permanent.</p>
            <ul className="mt-2 grid gap-2">{groups.unconfirmed.map(classCard)}</ul>
          </section> : null}
          {groups.candidate.length > 0 ? <section aria-label="Nouvelles classes confirmées">
            <h3 className="ui-title-sm">Nouvelles confirmées <span className="ui-text-caption">({groups.candidate.length})</span></h3>
            <p className="ui-text-caption">Utilisables par le prochain candidat, sans activation automatique.</p>
            <ul className="mt-2 grid gap-2">{groups.candidate.map(classCard)}</ul>
          </section> : null}
          {groups.active.length > 0 ? <details open={query.trim() ? true : undefined} className="border-t border-[color:var(--border-subtle)] pt-3">
            <summary className="cursor-pointer ui-title-sm">Classes du modèle actif ({groups.active.length})</summary>
            <ul className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">{groups.active.map(classCard)}</ul>
          </details> : null}
          {groups.unconfirmed.length + groups.candidate.length + groups.active.length === 0 ? <p className="ui-empty-state p-3">{query.trim() ? "Aucune classe ne correspond à cette recherche." : "Aucune classe disponible."}</p> : null}
        </div>
      )}
    </section>
  );
}
