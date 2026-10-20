import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";
import { deleteAnalysis, getHistory, getLegacyImportCount, importLegacyHistory } from "../../services/storage";
import type { AnalysisRecord } from "../../types";
import {
  getWorkspaceElementClassName,
  hasWorkspaceSubmittedAnnotation,
} from "./workspaceViewUtils";

export function resolveCurrentRecord(
  records: AnalysisRecord[],
  preferredId?: string | null,
) {
  if (preferredId) {
    return (
      records.find((record) => record.id === preferredId) ?? records[0] ?? null
    );
  }

  return records[0] ?? null;
}

export function useWorkspaceHistory() {
  const [searchParams] = useSearchParams();
  const initialPreferredId = searchParams.get("analysis");
  const initializedRef = useRef(false);

  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [currentRecord, setCurrentRecord] = useState<AnalysisRecord | null>(null);
  const [filter, setFilter] = useState("");
  const [historyOpen, setHistoryOpen] = useState(true);
  const [storageLoading, setStorageLoading] = useState(true);
  const [legacyImportCount, setLegacyImportCount] = useState(0);
  const [legacyImportError, setLegacyImportError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [storageError, setStorageError] = useState<string | null>(null);

  const selectRecord = (record: AnalysisRecord | null) => {
    setCurrentRecord(record);
  };

  const syncRecords = useCallback(async (preferredId?: string | null) => {
    try {
      const nextRecords = await getHistory();
      setRecords(nextRecords);
      selectRecord(resolveCurrentRecord(nextRecords, preferredId));
      setStorageError(null);
      return nextRecords;
    } catch (issue) {
      setStorageError("Stockage local indisponible. Vos analyses n'ont pas été effacées.");
      throw issue;
    }
  }, []);

  useEffect(() => {
    let active = true;

    getHistory()
      .then((nextRecords) => {
        if (!active) return;
        setStorageError(null);
        setRecords(nextRecords);
        setCurrentRecord(resolveCurrentRecord(nextRecords, initialPreferredId));
        if (!initializedRef.current) {
          setHistoryOpen(!resolveCurrentRecord(nextRecords, initialPreferredId));
          initializedRef.current = true;
        }
      })
      .catch(() => {
        if (active) setStorageError("Stockage local indisponible. Vos analyses n'ont pas été effacées.");
      })
      .finally(() => {
        if (active) setStorageLoading(false);
      });

    return () => {
      active = false;
    };
  }, [initialPreferredId]);

  const refreshLegacyImportCount = useCallback(() => getLegacyImportCount().then((count) => {
      setLegacyImportCount(count);
      setLegacyImportError(null);
    }).catch(() => {
      setLegacyImportError("Ancien historique inaccessible. Vérifiez le stockage du navigateur et réessayez.");
    }), []);

  useEffect(() => { void refreshLegacyImportCount(); }, [refreshLegacyImportCount]);

  const importLegacy = async () => {
    try {
      await importLegacyHistory();
      await syncRecords();
      setLegacyImportCount(0);
      setLegacyImportError(null);
    } catch {
      setLegacyImportError("L'import a échoué. Les anciennes analyses sont conservées ; réessayez.");
    }
  };

  const removeRecord = async (id: string) => {
    const name = records.find((record) => record.id === id)?.imageName ?? "cette analyse";
    if (!window.confirm(`Supprimer définitivement « ${name} » de l’historique ?`)) return;
    setHistoryError(null);
    try {
      await deleteAnalysis(id);
      await syncRecords(currentRecord?.id === id ? null : currentRecord?.id);
    } catch {
      setHistoryError("La suppression de l’analyse a échoué. Elle est conservée dans l’historique.");
    }
  };

  const filteredRecords = useMemo(() => {
    if (!filter) {
      return records;
    }

    const query = filter.toLowerCase();
    return records.filter((record) => {
      const names = record.result.elements.map((element) =>
        element.class_name.toLowerCase(),
      );
      const annotated = Object.values(record.annotations ?? {}).map((value) =>
        value.toLowerCase(),
      );
      return (
        names.some((name) => name.includes(query)) ||
        annotated.some((annotation) => annotation.includes(query)) ||
        record.imageName.toLowerCase().includes(query)
      );
    });
  }, [filter, records]);

  const stats = useMemo(() => {
    if (!currentRecord) {
      return null;
    }

    const elements = currentRecord.result.elements;
    const rejectedCount = elements.filter((element) => element.rejected).length;
    const classCounts: Record<string, number> = {};

    elements.forEach((_element, idx) => {
      const finalClass = getWorkspaceElementClassName(currentRecord, idx).trim();
      if (!finalClass) return;
      classCounts[finalClass] = (classCounts[finalClass] || 0) + 1;
    });

    let topClass = "None";
    let maxCount = 0;
    Object.entries(classCounts).forEach(([className, count]) => {
      if (count > maxCount) {
        maxCount = count;
        topClass = className;
      }
    });

    const annotatedCount = elements.filter((_element, idx) =>
      hasWorkspaceSubmittedAnnotation(currentRecord, idx),
    ).length;
    const topClasses = Object.entries(classCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([className, count]) => `${className} ${count}`);

    return {
      total: elements.length,
      rejectedCount,
      annotatedCount,
      submittedCount: annotatedCount,
      topClass,
      topClasses,
      imageSizeLabel: `${currentRecord.result.image_size[0]}×${currentRecord.result.image_size[1]}`,
    };
  }, [currentRecord]);

  return {
    records,
    currentRecord,
    filter,
    historyOpen,
    filteredRecords,
    stats,
    storageLoading,
    legacyImportCount,
    legacyImportError,
    historyError,
    storageError,
    importLegacy,
    refreshLegacyImportCount,
    setFilter,
    setHistoryOpen,
    selectRecord,
    syncRecords,
    removeRecord,
  };
}
