import type { AnalysisRecord, AnnotationStatus } from '../types';
import {
  createIndexedDbStorageDb,
  type AnalysisStorageDb,
} from './storageDb';
import {
  hasLegacyPayload,
  LEGACY_MIGRATION_MARKER_KEY,
  markLegacyMigrationComplete,
  readLegacyAnalysisRecords,
} from './storageLegacy';

export type StorageInitResult = {
  ok: boolean;
  migrated: boolean;
  legacyFound: boolean;
  legacyRecordCount: number;
  importedCount: number;
  error?: Error;
};

type StorageDbFactory = (name: string) => AnalysisStorageDb;

type StorageTestOptions = {
  dbFactory?: StorageDbFactory;
};

let dbFactory: StorageDbFactory = (name) => createIndexedDbStorageDb({ dbName: name });
let db: AnalysisStorageDb | null = null;
let initPromise: Promise<StorageInitResult> | null = null;
let lastInitResult: StorageInitResult | null = null;
let activeAccountId: string | null = 'local';
let accountGeneration = 0;
let legacyImportAllowed = true;
const LEGACY_OWNER_KEY = 'clinic-codex-legacy-owner';

export function setStorageAccount(accountId: string | null, allowLegacyImport = false): void {
  legacyImportAllowed = Boolean(accountId) && (accountId === 'local' || allowLegacyImport);
  if (accountId === activeAccountId) return;
  db?.close();
  db = null;
  initPromise = null;
  lastInitResult = null;
  activeAccountId = accountId;
  accountGeneration += 1;
}

function normalizeAnnotationStatus(status: unknown, elementCount: number): Record<number, AnnotationStatus> {
  if (!status || typeof status !== 'object') {
    return {};
  }

  const normalized: Record<number, AnnotationStatus> = {};
  Object.entries(status as Record<string, unknown>).forEach(([key, value]) => {
    const idx = Number(key);
    if (!Number.isInteger(idx) || idx < 0 || idx >= elementCount) return;
    if (value === 'validated' || value === 'draft') {
      normalized[idx] = value;
    }
  });
  return normalized;
}

export function normalizeAnalysisRecord(record: AnalysisRecord): AnalysisRecord {
  const elements = record.result?.elements ?? [];
  return {
    ...record,
    annotations: record.annotations ?? {},
    annotationStatus: normalizeAnnotationStatus(record.annotationStatus, elements.length),
    result: {
      ...record.result,
      num_elements: elements.length,
      elements,
    },
  };
}

function toError(issue: unknown) {
  return issue instanceof Error ? issue : new Error(String(issue));
}

function getDb() {
  if (!activeAccountId) throw new Error('Connectez-vous pour accéder à l’historique local');
  if (!db) {
    db = dbFactory(`clinic-codex-storage-account-${encodeURIComponent(activeAccountId)}`);
  }
  return db;
}

function legacyHistory() {
  return readLegacyAnalysisRecords().map(normalizeAnalysisRecord);
}

export async function initializeStorage(): Promise<StorageInitResult> {
  if (initPromise) return initPromise;

  const pending = initPromise = (async () => {
    const legacyFound = legacyImportAllowed && hasLegacyPayload();
    const generation = accountGeneration;

    try {
      const storageDb = getDb();
      await storageDb.listRecords();
      if (generation !== accountGeneration) throw new Error('Le compte actif a changé');
      const result: StorageInitResult = {
        ok: true,
        migrated: false,
        legacyFound,
        legacyRecordCount: 0,
        importedCount: 0,
      };
      lastInitResult = result;
      return result;
    } catch (issue) {
      if (generation === accountGeneration) {
        db?.close();
        db = null;
        initPromise = null;
      }
      const result: StorageInitResult = {
        ok: false,
        migrated: false,
        legacyFound,
        legacyRecordCount: 0,
        importedCount: 0,
        error: toError(issue),
      };
      if (generation === accountGeneration) lastInitResult = result;
      return result;
    }
  })();

  return pending.then((result) => {
    if (!result.ok && initPromise === pending) initPromise = null;
    return result;
  });
}

async function getWritableDb() {
  const generation = accountGeneration;
  const result = await initializeStorage();
  if (!result.ok || generation !== accountGeneration) {
    throw result.error ?? new Error('Browser storage is unavailable');
  }
  return getDb();
}

export function getLastStorageInitResult() {
  return lastInitResult;
}

export async function getHistory(): Promise<AnalysisRecord[]> {
  const generation = accountGeneration;
  const result = await initializeStorage();
  if (generation !== accountGeneration) return [];
  if (!result.ok) throw result.error ?? new Error('Browser storage is unavailable');

  try {
    const records = await getDb().listRecords();
    if (generation !== accountGeneration) return [];
    return records.map((stored) => normalizeAnalysisRecord(stored.record));
  } catch (issue) {
    if (generation !== accountGeneration) return [];
    lastInitResult = { ...result, ok: false, error: toError(issue) };
    db?.close();
    db = null;
    initPromise = null;
    throw lastInitResult.error;
  }
}

export async function getAnalysisById(id: string): Promise<AnalysisRecord | null> {
  const generation = accountGeneration;
  const result = await initializeStorage();
  if (generation !== accountGeneration) return null;
  if (!result.ok) throw result.error ?? new Error('Browser storage is unavailable');

  try {
    const stored = await getDb().getRecord(id);
    if (generation !== accountGeneration) return null;
    return stored ? normalizeAnalysisRecord(stored.record) : null;
  } catch (issue) {
    if (generation !== accountGeneration) return null;
    lastInitResult = { ...result, ok: false, error: toError(issue) };
    db?.close();
    db = null;
    initPromise = null;
    throw lastInitResult.error;
  }
}

async function legacyRecords(): Promise<AnalysisRecord[]> {
  let stored: Awaited<ReturnType<AnalysisStorageDb['listRecords']>> = [];
  let previousDb: AnalysisStorageDb | null = null;
  if (globalThis.indexedDB) {
    try {
      previousDb = createIndexedDbStorageDb();
      stored = await previousDb.listRecords();
    } finally {
      previousDb?.close();
    }
  }
  const found = new Map<string, AnalysisRecord>();
  for (const item of stored) found.set(item.id, normalizeAnalysisRecord(item.record));
  for (const record of legacyHistory()) {
    if (!found.has(record.id)) found.set(record.id, record);
  }
  return [...found.values()];
}

export async function getLegacyImportCount(): Promise<number> {
  if (!legacyImportAllowed) return 0;
  const owner = localStorage.getItem(LEGACY_OWNER_KEY);
  if (owner && owner !== activeAccountId) return 0;
  if (owner && localStorage.getItem(LEGACY_MIGRATION_MARKER_KEY)) return 0;
  const generation = accountGeneration;
  const count = (await legacyRecords()).length;
  return generation === accountGeneration ? count : 0;
}

export async function importLegacyHistory(): Promise<number> {
  if (!legacyImportAllowed || !activeAccountId) throw new Error('Import réservé à l’administration locale');
  const owner = localStorage.getItem(LEGACY_OWNER_KEY);
  if (owner && owner !== activeAccountId) throw new Error('Cet historique a déjà été attribué à un autre compte');
  // Reserve unowned data during import, but release the reservation if import fails.
  const claimant = activeAccountId;
  const claimed = !owner;
  if (claimed) localStorage.setItem(LEGACY_OWNER_KEY, claimant);
  const generation = accountGeneration;
  try {
    const previous = await legacyRecords();
    const storageDb = await getWritableDb();
    if (generation !== accountGeneration) throw new Error('Le compte actif a changé');
    const count = await storageDb.importMissingRecords(previous);
    markLegacyMigrationComplete(count);
    return count;
  } catch (issue) {
    if (claimed && localStorage.getItem(LEGACY_OWNER_KEY) === claimant && !localStorage.getItem(LEGACY_MIGRATION_MARKER_KEY)) {
      localStorage.removeItem(LEGACY_OWNER_KEY);
    }
    throw issue;
  }
}

export async function saveAnalysis(record: AnalysisRecord): Promise<void> {
  const storageDb = await getWritableDb();
  await storageDb.saveRecord(normalizeAnalysisRecord(record));
}

export async function updateAnnotations(id: string, annotations: Record<number, string>): Promise<boolean> {
  try {
    const storageDb = await getWritableDb();
    return storageDb.updateAnnotations(id, annotations);
  } catch {
    return false;
  }
}

export async function updateElements(
  id: string,
  elements: AnalysisRecord['result']['elements'],
  annotationStatus?: Record<number, AnnotationStatus>,
): Promise<boolean> {
  try {
    const storageDb = await getWritableDb();
    const existing = await storageDb.getRecord(id);
    if (!existing) return false;
    const nextStatus = normalizeAnnotationStatus(
      annotationStatus ?? existing.record.annotationStatus,
      elements.length,
    );
    return storageDb.updateElements(id, elements, nextStatus);
  } catch {
    return false;
  }
}

export async function deleteAnalysis(id: string): Promise<void> {
  const storageDb = await getWritableDb();
  await storageDb.deleteRecord(id);
}

export function __resetStorageForTests(options: StorageTestOptions = {}): void {
  db?.close();
  db = null;
  initPromise = null;
  lastInitResult = null;
  activeAccountId = 'local';
  legacyImportAllowed = true;
  accountGeneration += 1;
  dbFactory = options.dbFactory ?? ((name) => createIndexedDbStorageDb({ dbName: name }));
}
