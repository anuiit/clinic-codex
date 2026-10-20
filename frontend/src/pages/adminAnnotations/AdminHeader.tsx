import styles from "./AdminHeader.module.css";
import { useMemo } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import ThemeToggle from "../../components/ThemeToggle";
import { ActionButton, PageTabs } from "../../components/ui/AdminPrimitives";
import type { AdminAnnotationQueue } from "../../types";
import { ADMIN_TABS, type AdminAnnotationsPageProps, type AdminTab, formatTimestamp } from "./model";
import { RuntimeVersionBadge } from "../../components/RuntimeVersionBadge";

export function AdminHeader({
  themeMode = "dark",
  onToggleTheme,
  queue,
  canReadQueue = true,
  canReadTraining = true,
  refreshing,
  refreshDisabled,
  refreshDisabledReason,
  lastRefreshedAt,
  onRefresh,
  activeTab,
  onSelectTab,
  authSlot,
  onReturnToAnalysis,
}: AdminAnnotationsPageProps & {
  queue: AdminAnnotationQueue | null;
  refreshing: boolean;
  refreshDisabled: boolean;
  refreshDisabledReason: string;
  lastRefreshedAt: Date | null;
  onRefresh: () => void;
  activeTab: AdminTab;
  onReturnToAnalysis: (event: ReactMouseEvent<HTMLAnchorElement>) => void;
  onSelectTab: (tab: AdminTab) => void;
}) {
  return (
    <header role="banner" className={`${styles.owner} admin-command-bar admin-chrome`}>
      <div className="admin-command-main">
        <div className="admin-command-title admin-brand">
          <h1 className="text-base font-semibold tracking-tight text-[color:var(--text-heading)]">Poste de triage</h1>
        </div>
        <AdminTabs activeTab={activeTab} onSelect={onSelectTab} queue={queue} canReadQueue={canReadQueue} canReadTraining={canReadTraining} />
        <div className="admin-command-actions">
          <a href="/" onClick={onReturnToAnalysis} className="px-2 py-1 text-xs underline underline-offset-2">
            Retour à l’analyse
          </a>
          {authSlot}
          <ActionButton
            tone="ghost"
            className="px-2.5 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-50"
            disabled={refreshDisabled}
            onClick={onRefresh}
            title={refreshDisabled ? refreshDisabledReason : "Actualiser la file"}
          >
            {refreshing ? "Actualisation…" : "Actualiser"}
          </ActionButton>
          <details className="admin-options">
            <summary>Options du poste</summary>
            <div className="admin-options-panel">
              <RuntimeVersionBadge />
              <span className="admin-timestamp">Actualisé {formatTimestamp(lastRefreshedAt)}</span>
              {onToggleTheme ? <ThemeToggle mode={themeMode} onToggle={onToggleTheme} className="shrink-0" /> : null}
            </div>
          </details>
        </div>
      </div>
    </header>
  );
}
function AdminTabs({
  activeTab,
  onSelect,
  queue,
  canReadQueue,
  canReadTraining,
}: {
  activeTab: AdminTab;
  onSelect: (tab: AdminTab) => void;
  queue: AdminAnnotationQueue | null;
  canReadQueue: boolean;
  canReadTraining: boolean;
}) {
  const tabItems = useMemo(() => {
    const availableTabs = ADMIN_TABS.filter((tab) =>
      tab.id === "training" || tab.id === "compare" ? canReadTraining : canReadQueue);
    if (!queue) {
      return availableTabs;
    }
    const reviewed = queue.counts.approved + queue.counts.rejected;
    return availableTabs.map((tab) => ({
      ...tab,
      hint:
        tab.id === "review"
          ? `${reviewed}/${queue.counts.total}`
          : tab.id === "dataset"
            ? queue.counts.trainable
            : tab.id === "classes" || tab.id === "compare"
              ? undefined
            : undefined,
    }));
  }, [queue, canReadQueue, canReadTraining]);
  return (
    <PageTabs
      items={tabItems}
      activeId={activeTab}
      onSelect={onSelect}
      ariaLabel="Étapes du poste de triage"
      panelIdPrefix="admin"
      variant="underline"
    />
  );
}
