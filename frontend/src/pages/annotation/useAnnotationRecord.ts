import { useEffect, useRef, useState } from "react";
import { getAnnotationClasses } from "../../services/api";
import { getAnalysisById } from "../../services/storage";
import type { AnalysisRecord, AnnotationStatus, DetectedElement } from "../../types";

export function cloneElements(elements: DetectedElement[]): DetectedElement[] {
  return elements.map((element) => ({
    ...element,
    bbox: [...element.bbox],
    top_k: element.top_k.map((item) => ({ ...item })),
  }));
}

export function useAnnotationRecord(id: string | undefined, initialFocusedIdx: number | null) {
  const cardRefs = useRef<Array<HTMLElement | null>>([]);
  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [elements, setElements] = useState<DetectedElement[]>([]);
  const [annotationStatus, setAnnotationStatus] = useState<
    Record<number, AnnotationStatus>
  >({});
  const [classes, setClasses] = useState<string[]>([]);
  const [customClasses, setCustomClasses] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [storageError, setStorageError] = useState(false);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [focusedIdx, setFocusedIdx] = useState<number | null>(initialFocusedIdx);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [listHoveredIdx, setListHoveredIdx] = useState<number | null>(null);

  useEffect(() => {
    let active = true;

    async function loadData() {
      setLoading(true);
      setRecord(null);
      setElements([]);
      setAnnotationStatus({});
      setStorageError(false);

      if (!id) {
        if (active) setLoading(false);
        return;
      }

      let rec: AnalysisRecord | null;
      try {
        rec = await getAnalysisById(id);
      } catch {
        if (active) {
          setStorageError(true);
          setLoading(false);
        }
        return;
      }
      if (!active) return;
      setRecord(rec);

      if (!rec) {
        setLoading(false);
        return;
      }
      setElements(cloneElements(rec.result.elements));
      setAnnotationStatus(rec.annotationStatus ?? {});
      setFocusedIdx(initialFocusedIdx !== null && initialFocusedIdx < rec.result.elements.length ? initialFocusedIdx : null);

      try {
        const classesResult = await getAnnotationClasses();
        if (active) setClasses(classesResult.class_names);
      } catch {
        // failed to load classes
      } finally {
        if (active) setLoading(false);
      }
    }
    void loadData();
    return () => {
      active = false;
    };
  }, [id, initialFocusedIdx, loadAttempt]);

  return {
    cardRefs,
    record,
    elements,
    annotationStatus,
    classes,
    customClasses,
    loading,
    storageError,
    retryLoad: () => setLoadAttempt((attempt) => attempt + 1),
    focusedIdx,
    hoveredIdx,
    listHoveredIdx,
    setElements,
    setAnnotationStatus,
    setCustomClasses,
    setFocusedIdx,
    setHoveredIdx,
    setListHoveredIdx,
  };
}
