import type { RefObject, ReactNode } from "react";
import { AlertCircle, Upload } from "lucide-react";
import { ThemeToggle, type ThemeMode } from "../../components/ThemeToggle";
import { RuntimeVersionBadge } from "../../components/RuntimeVersionBadge";
import adminHeaderStyles from "../adminAnnotations/AdminHeader.module.css";
import styles from "./WorkspaceChrome.module.css";

type WorkspaceHeaderProps = {
  inputRef: RefObject<HTMLInputElement | null>;
  loading: boolean;
  error: string | null;
  labels: { appTitle: string; uploadPrompt: string };
  authSlot?: ReactNode;
  onFileSelected: (file: File) => void;
  themeMode: ThemeMode;
  onToggleTheme: () => void;
};

export default function WorkspaceHeader({
  inputRef,
  loading,
  error,
  labels,
  authSlot,
  onFileSelected,
  themeMode,
  onToggleTheme,
}: WorkspaceHeaderProps) {
  return (
    <header role="banner" className={`${adminHeaderStyles.owner} ${styles.owner} admin-command-bar`}>
      <div className="admin-command-main">
        <h1 className="admin-command-title text-base font-semibold tracking-tight text-[color:var(--text-heading)]">
          {labels.appTitle}
        </h1>
        <div className="admin-command-actions">
          {authSlot}
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/bmp,image/*"
            className="hidden"
            tabIndex={-1}
            onChange={(event) => {
              const nextFile = event.target.files?.[0];
              if (nextFile) onFileSelected(nextFile);
            }}
          />
          <button
            type="button"
            disabled={loading}
            onClick={() => inputRef.current?.click()}
            className="ui-action-primary inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm disabled:opacity-50"
          >
            <Upload size={16} aria-hidden="true" />
            {labels.uploadPrompt}
          </button>
          <details className="admin-options">
            <summary>Options</summary>
            <div className="admin-options-panel">
              <RuntimeVersionBadge />
              <ThemeToggle mode={themeMode} onToggle={onToggleTheme} />
            </div>
          </details>
        </div>
      </div>
      {error ? (
        <div role="alert" className="ui-alert ui-alert--danger flex items-center gap-2 px-3 py-1.5 text-xs">
          <AlertCircle size={14} aria-hidden="true" />
          {error}
        </div>
      ) : null}
    </header>
  );
}
