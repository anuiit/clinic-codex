import type { DetectedElement } from "../../types";
import { isUnnamedClass } from "../../utils/fuzzyClasses";

export function formatBboxLabel(
  idx: number,
  className: string,
  showName: boolean,
  unnamedLabel: string,
): string {
  if (!showName) return `#${idx}`;
  const displayName = isUnnamedClass(className) ? unnamedLabel : className;
  return displayName.length > 18 ? `${displayName.slice(0, 17)}…` : displayName;
}

export function clearApproval(element: DetectedElement): DetectedElement {
  const next: DetectedElement = { ...element, source: "manual_edit", confidence_kind: "unscored",
    confidence: 0, class_label: null, top_k: [] };
  delete next.review;
  return next;
}
