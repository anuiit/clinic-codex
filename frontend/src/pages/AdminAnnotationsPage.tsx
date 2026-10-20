import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import sharedStyles from "./adminAnnotations/AdminShared.module.css";
import "./AdminAnnotationsPage.css";
import { AdminHeader } from "./adminAnnotations/AdminHeader";
import { DatasetTab } from "./adminAnnotations/DatasetTab";
import { ClassesTab } from "./adminAnnotations/ClassesTab";
import { CompareTab } from "./adminAnnotations/CompareTab";
import { ReviewTab } from "./adminAnnotations/ReviewTab";
import { TrainingTab } from "./adminAnnotations/TrainingTab";
import { PanelSkeleton } from "./adminAnnotations/shared";
import { AnnotationToast } from "./annotation/AnnotationToast";
import { UnsavedNavigationPrompt } from "../components/UnsavedNavigationPrompt";
import { ADMIN_QUEUE_AUTO_REFRESH_MS, ADMIN_TABS, STATUS_LABEL, type AdminAnnotationsPageProps, type AdminTab } from "./adminAnnotations/model";
import {
  getAdminAnnotationQueue,
  getAdminClasses,
  modifyAdminAnnotationElement,
  restoreAdminAnnotationElement,
  setAdminAnnotationReviewStatus,
} from "../services/api";
import type {
  AdminAnnotationElement,
  AdminAnnotationModifyPayload,
  AdminAnnotationQueue,
  AdminAnnotationReviewStatus,
} from "../types";

function AdminAnnotationsPage({
  themeMode = "dark",
  onToggleTheme,
  initialTab = "review",
  onNavigateTab,
  onCompareCandidate,
  comparisonVersionId,
  authSlot,
  canReadQueue = true,
  canReadTraining = true,
  canRunTraining = true,
  guardRouteTransitions = false,
  canReview = true,
}: AdminAnnotationsPageProps) {
  const [queue, setQueue] = useState<AdminAnnotationQueue | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [mutatingKey, setMutatingKey] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editingDirty, setEditingDirty] = useState(false);
  const allowLeaveRef = useRef(false);
  const [classNames, setClassNames] = useState<string[]>([]);
  const [reviewClassFilter, setReviewClassFilter] = useState<string | null>(null);
  const [localComparisonVersion, setLocalComparisonVersion] = useState<string | undefined>(comparisonVersionId);
  const [localActiveTab, setLocalActiveTab] = useState<AdminTab>(initialTab);
  const activeTab = onNavigateTab ? initialTab : localActiveTab;
  useEffect(() => { allowLeaveRef.current = false; }, [activeTab]);
  const [queueRefreshing, setQueueRefreshing] = useState(false);
  const [lastQueueRefreshAt, setLastQueueRefreshAt] = useState<Date | null>(
    null,
  );

  useEffect(() => {
    if (!actionMessage) return;
    const timeout = window.setTimeout(() => setActionMessage(null), 5000);
    return () => window.clearTimeout(timeout);
  }, [actionMessage]);

  const syncQueue = useCallback(async () => {
    if (!canReadQueue) return null;
    const nextQueue = await getAdminAnnotationQueue();
    setQueue(nextQueue);
    setLastQueueRefreshAt(new Date());
    setActionError(null);
    return nextQueue;
  }, [canReadQueue]);

  const handleSelectTab = useCallback(
    (tab: AdminTab) => {
      if (editingDirty && !window.confirm("La correction en cours n'est pas enregistrée. Quitter quand même ?")) return;
      allowLeaveRef.current = true;
      setActionMessage(null);
      setEditingKey(null);
      setEditingDirty(false);
      if (onNavigateTab) {
        onNavigateTab(tab);
        return;
      }
      setLocalActiveTab(tab);
      allowLeaveRef.current = false;
    },
    [editingDirty, onNavigateTab],
  );

  useEffect(() => {
    if (!editingDirty) return;
    const warn = (event: BeforeUnloadEvent) => {
      if (allowLeaveRef.current) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [editingDirty]);

  const loadQueue = useCallback(
    async ({ showLoading = true }: { showLoading?: boolean } = {}) => {
      if (showLoading) {
        setLoading(true);
      } else {
        setQueueRefreshing(true);
      }
      setLoadError(null);
      try {
        await syncQueue();
      } catch {
        setLoadError("Impossible de charger la file locale de triage.");
      } finally {
        if (showLoading) {
          setLoading(false);
        } else {
          setQueueRefreshing(false);
        }
      }
    },
    [syncQueue],
  );

  useEffect(() => {
    if (!canReadQueue) return;
    const timeout = window.setTimeout(() => {
      void loadQueue();
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [loadQueue, canReadQueue]);

  useEffect(() => {
    let cancelled = false;
    if (!canReadQueue) return;
    void (async () => {
      try {
        const classes = await getAdminClasses();
        if (!cancelled) {
          setClassNames(classes.classes.filter((item) => item.status !== "unconfirmed").map((item) => item.class_name));
        }
      } catch {
        if (!cancelled) {
          setClassNames([]);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [canReadQueue]);

  const canRefreshQueue =
    canReadQueue && !loading && !queueRefreshing && !mutatingKey && !editingKey;
  const refreshDisabledReason = editingKey
    ? "Terminez ou annulez la correction avant d'actualiser la file."
    : mutatingKey
      ? "Attendez la fin de l'action de triage avant d'actualiser."
      : loading || queueRefreshing
        ? "Actualisation déjà en cours."
        : "Actualiser la file";

  const handleManualRefresh = useCallback(() => {
    if (!canRefreshQueue) {
      return;
    }
    void loadQueue({ showLoading: false });
  }, [canRefreshQueue, loadQueue]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (document.visibilityState === "hidden" || !canRefreshQueue) {
        return;
      }
      void loadQueue({ showLoading: false });
    }, ADMIN_QUEUE_AUTO_REFRESH_MS);
    return () => window.clearInterval(interval);
  }, [canRefreshQueue, loadQueue]);

  const handleSelectElement = useCallback((element: AdminAnnotationElement) => {
    if (element.key === selectedKey) return;
    if (editingDirty && element.key !== selectedKey &&
        !window.confirm("La correction en cours n'est pas enregistrée. Changer d'élément ?")) return;
    setSelectedKey(element.key);
    setEditingKey((current) => (current === element.key ? current : null));
    setEditingDirty(false);
  }, [editingDirty, selectedKey]);

  const mutationFailure = async (error: unknown, fallback: string) => {
    if (axios.isAxiosError(error) && error.response?.data?.error_code === "REVIEW_MIGRATION_REQUIRED") {
      setActionError("Les anciennes validations doivent être importées avant toute modification. Aucune annotation n'a été perdue.");
      return;
    }
    if (axios.isAxiosError(error) && error.response?.status === 409) {
      try {
        await syncQueue();
        setActionError("Cet élément a changé depuis son chargement. La file a été actualisée : vérifiez la version affichée avant de réessayer.");
      } catch {
        setActionError("Conflit de modification. Actualisez la file avant de réessayer.");
      }
    } else {
      setActionError(fallback);
    }
  };

  const handleReview = async (
    element: AdminAnnotationElement,
    status: AdminAnnotationReviewStatus,
  ) => {
    setActionError(null);
    setActionMessage(null);
    setSelectedKey(element.key);
    setMutatingKey(element.key);
    let committed = false;
    try {
      await setAdminAnnotationReviewStatus(
        element.analysis_id,
        element.index,
        status,
        element.revision,
      );
      committed = true;
      try {
        await syncQueue();
        setActionMessage(
          `${STATUS_LABEL[status]} · ${element.class_name} · ${queue?.analyses.find((analysis) => analysis.analysis_id === element.analysis_id)?.image_name || element.analysis_id}`,
        );
      } catch {
        setActionError(
          `Décision enregistrée pour ${element.class_name}, mais la file n'a pas pu être actualisée. Utilisez Actualiser pour resynchroniser.`,
        );
      }
    } catch (error) {
      await mutationFailure(error, `Impossible de marquer l'élément ${element.index} comme ${STATUS_LABEL[status]}.`);
    } finally {
      setMutatingKey(null);
    }
    return committed;
  };

  const handleModify = async (
    element: AdminAnnotationElement,
    payload: AdminAnnotationModifyPayload,
  ) => {
    setActionError(null);
    setActionMessage(null);
    setSelectedKey(element.key);
    setMutatingKey(element.key);
    try {
      await modifyAdminAnnotationElement(
        element.analysis_id,
        element.index,
        payload,
      );
      const successMessage = payload.approve_after_save
        ? `Élément ${element.index} enregistré et validé.`
        : `Changements de l'élément ${element.index} enregistrés.`;
      try {
        await syncQueue();
        setActionMessage(successMessage);
      } catch {
        setActionError(`${successMessage} La file n'a pas pu être actualisée ; utilisez Actualiser pour resynchroniser.`);
      }
      if (payload.approve_after_save) {
        setEditingKey(null);
      }
      setEditingDirty(false);
    } catch (error) {
      await mutationFailure(error, `Impossible d'enregistrer les changements de l'élément ${element.index}.`);
    } finally {
      setMutatingKey(null);
    }
  };

  const handleRestore = async (element: AdminAnnotationElement, targetRevision: number) => {
    setActionError(null);
    setActionMessage(null);
    setMutatingKey(element.key);
    try {
      await restoreAdminAnnotationElement(element.analysis_id, element.index, targetRevision, element.revision);
      let refreshed = true;
      try {
        await syncQueue();
      } catch {
        refreshed = false;
      }
      setEditingKey(null);
      setEditingDirty(false);
      const message = `Version ${targetRevision} restaurée pour l'élément ${element.index}. À vérifier avant validation.`;
      if (refreshed) setActionMessage(message);
      else setActionError(`${message} La file n'a pas pu être actualisée ; utilisez Actualiser pour resynchroniser.`);
    } catch (error) {
      await mutationFailure(error, "Impossible de restaurer cette version.");
    } finally {
      setMutatingKey(null);
    }
  };

  const queueElementKeys =
    queue?.analyses.flatMap((analysis) =>
      analysis.elements.map((element) => element.key),
    ) ?? [];
  const effectiveSelectedKey =
    selectedKey && queueElementKeys.includes(selectedKey)
      ? selectedKey
      : (queueElementKeys[0] ?? null);
  const effectiveEditingKey =
    queue?.review_store?.mode !== "legacy_readonly" && editingKey && queueElementKeys.includes(editingKey) ? editingKey : null;

  return (
    <div
      className={`${sharedStyles.owner} admin-console flex h-full min-h-0 flex-col overflow-hidden`}
      data-theme={themeMode}
      onClickCapture={(event) => {
        if (!editingDirty || !(event.target instanceof Element) || !event.target.closest("[data-auth-logout]")) return;
        if (!window.confirm("La correction en cours n'est pas enregistrée. Se déconnecter quand même ?")) {
          event.preventDefault();
          event.stopPropagation();
        }
      }}
    >
      {guardRouteTransitions ? <UnsavedNavigationPrompt dirty={editingDirty} allowLeaveRef={allowLeaveRef} message="La correction en cours n'est pas enregistrée. Quitter quand même ?" /> : null}
      <AdminHeader
        themeMode={themeMode}
        onToggleTheme={onToggleTheme}
        queue={queue}
        canReadQueue={canReadQueue}
        canReadTraining={canReadTraining}
        refreshing={queueRefreshing}
        refreshDisabled={!canRefreshQueue}
        refreshDisabledReason={refreshDisabledReason}
        lastRefreshedAt={lastQueueRefreshAt}
        onRefresh={handleManualRefresh}
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        authSlot={authSlot}
        onReturnToAnalysis={(event) => {
          if (event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) return;
          if (editingDirty && !window.confirm("La correction en cours n'est pas enregistrée. Revenir à l'analyse ?")) {
            event.preventDefault();
          } else {
            allowLeaveRef.current = true;
          }
        }}
      />

      <div className="admin-alert-stack">
        {loading && canReadQueue ? <PanelSkeleton label="Chargement de la file de triage" /> : null}

        {loadError ? (
          <div role="alert" className="ui-alert ui-alert--danger p-3">
            {loadError}
          </div>
        ) : null}

        {queue?.review_store?.mode === "legacy_readonly" ? (
          <div role="alert" className="ui-alert ui-alert--accent p-3">
            <strong>Anciennes validations à importer</strong>
            <p>{queue.review_store.legacy_decisions} validation{queue.review_store.legacy_decisions > 1 ? "s" : ""} conservée{queue.review_store.legacy_decisions > 1 ? "s" : ""} en lecture seule. L’administrateur doit sauvegarder les annotations, lancer l’import local puis redémarrer l’application. Les décisions restent visibles, mais aucune modification n’est possible avant l’import.</p>
          </div>
        ) : null}

        {actionError ? (
          <div role="alert" className="ui-alert ui-alert--danger p-3">
            {actionError}
          </div>
        ) : null}

      </div>

      {actionMessage ? <AnnotationToast toast={{ msg: actionMessage, ok: true }} /> : null}

      {canReadQueue || activeTab === "training" || activeTab === "compare" ? (
        <div
          className={`admin-tab-content ${activeTab === "review" ? "admin-tab-content--review" : ""}`}
        >
          {ADMIN_TABS.map((tab) => (
            <section
              key={tab.id}
              id={`admin-${tab.id}-panel`}
              role="tabpanel"
              aria-label={`Panneau ${tab.label}`}
              hidden={activeTab !== tab.id}
            >
              {queue && activeTab === tab.id && tab.id === "review" ? (
                <ReviewTab
                  queue={queue}
                  readOnly={!canReview || queue.review_store?.mode === "legacy_readonly"}
                  selectedKey={effectiveSelectedKey}
                  mutatingKey={mutatingKey}
                  editingKey={effectiveEditingKey}
                  onSelect={handleSelectElement}
                  classNames={classNames}
                  initialClassFilter={reviewClassFilter}
                  onReview={handleReview}
                  onModify={handleModify}
                  onRestore={handleRestore}
                  onEditingDirtyChange={setEditingDirty}
                  onEdit={(element) => {
                    setSelectedKey(element.key);
                    setEditingKey(element.key);
                  }}
                  onAnnulerEdit={() => { setEditingKey(null); setEditingDirty(false); }}
                />
              ) : null}
              {!queue && !loading && canReadQueue && activeTab === tab.id && (tab.id === "review" || tab.id === "dataset") ? (
                <div role="status" className="p-4">
                  <h2 className="ui-title-md">File de triage indisponible</h2>
                  <p className="mt-2">Les éléments ne peuvent pas être affichés pour le moment.</p>
                  <button type="button" className="ui-action-ghost mt-3 px-3 py-2" onClick={() => void loadQueue()}>
                    Réessayer
                  </button>
                </div>
              ) : null}
              {queue && activeTab === tab.id && tab.id === "dataset" ? (
                <DatasetTab
                  queue={queue}
                  onJumpToReview={(element) => {
                    handleSelectTab("review");
                    setSelectedKey(element.key);
                    setEditingKey(null);
                  }}
                />
              ) : null}
              {activeTab === tab.id && tab.id === "classes" ? (
                <ClassesTab
                  queue={queue}
                  canConfirm={canReview && queue?.review_store?.mode !== "legacy_readonly"}
                  onConfirmed={setClassNames}
                  onViewAnnotations={(className) => {
                    setReviewClassFilter(className);
                    handleSelectTab("review");
                  }}
                />
              ) : null}
              {activeTab === tab.id && tab.id === "training" ? (
                <TrainingTab canRunTraining={canRunTraining} onCompareCandidate={(versionId) => {
                  setLocalComparisonVersion(versionId);
                  if (onCompareCandidate) onCompareCandidate(versionId);
                  else handleSelectTab("compare");
                }} />
              ) : null}
              {activeTab === tab.id && tab.id === "compare" ? (
                <CompareTab initialVersionId={comparisonVersionId ?? localComparisonVersion} queue={queue} canRunTraining={canRunTraining} />
              ) : null}
            </section>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default AdminAnnotationsPage;
