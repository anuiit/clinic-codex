import type { AnalysisRecord } from '../types';

const STORAGE_KEY = 'codex_analyses';

export function saveAnalysis(record: AnalysisRecord): void {
  const history = getHistory();
  history.unshift(record);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
}

export function getHistory(): AnalysisRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as AnalysisRecord[]) : [];
  } catch {
    return [];
  }
}

export function getAnalysisById(id: string): AnalysisRecord | null {
  return getHistory().find((r) => r.id === id) ?? null;
}

export function updateAnnotations(id: string, annotations: Record<number, string>): boolean {
  const history = getHistory();
  const idx = history.findIndex((r) => r.id === id);
  if (idx === -1) return false;
  try {
    history[idx] = { ...history[idx], annotations };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    return true;
  } catch {
    return false;
  }
}

export function deleteAnalysis(id: string): void {
  const filtered = getHistory().filter((r) => r.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
}
