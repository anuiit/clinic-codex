import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AnalysisRecord } from '../types';
import { createMemoryStorageDb, type AnalysisStorageDb } from './storageDb';
import { LEGACY_MIGRATION_MARKER_KEY, LEGACY_STORAGE_KEY } from './storageLegacy';
import {
  __resetStorageForTests,
  deleteAnalysis,
  getAnalysisById,
  getHistory,
  getLegacyImportCount,
  getLastStorageInitResult,
  importLegacyHistory,
  initializeStorage,
  saveAnalysis,
  setStorageAccount,
  updateElements,
} from './storage';

const RECORD: AnalysisRecord = {
  id: 'analysis-id',
  imageName: 'sample.png',
  imageDataUrl: 'data:image/png;base64,abc',
  timestamp: 1704067200000,
  result: {
    num_elements: 2,
    image_size: [800, 600],
    elements: [
      { bbox: [1, 2, 3, 4], class_name: 'atl', class_label: 1, confidence: 0.9, rejected: false, top_k: [] },
      { bbox: [5, 6, 7, 8], class_name: 'beta', class_label: 2, confidence: 0.8, rejected: false, top_k: [] },
    ],
  },
  annotations: {},
};

const SECOND_RECORD: AnalysisRecord = {
  ...RECORD,
  id: 'second-id',
  imageName: 'second.png',
  timestamp: 1704153600000,
};

function cloneRecord(record: AnalysisRecord): AnalysisRecord {
  return structuredClone(record) as AnalysisRecord;
}

function resetWithDb(db: AnalysisStorageDb = createMemoryStorageDb()) {
  __resetStorageForTests({ dbFactory: () => db });
  return db;
}

describe('storage IndexedDB boundary compatibility', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    resetWithDb();
  });

  it('getHistory returns [] for missing legacy payload and empty IndexedDB', async () => {
    await expect(getHistory()).resolves.toEqual([]);
  });

  it('getHistory returns [] for invalid legacy JSON', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, '{not json');

    await expect(getHistory()).resolves.toEqual([]);
  });

  it('getHistory returns [] for valid JSON that is not an array', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify({ id: 'not-array' }));

    await expect(getHistory()).resolves.toEqual([]);
  });

  it('only imports old records after explicit consent, normalizing their annotation status', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));

    await expect(getHistory()).resolves.toEqual([]);
    await expect(getLegacyImportCount()).resolves.toBe(1);
    await expect(importLegacyHistory()).resolves.toBe(1);
    expect((await getHistory())[0]).toEqual(expect.objectContaining({ annotationStatus: {} }));
  });

  it('retains legacy localStorage and writes a marker only after explicit import', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));

    await initializeStorage();
    expect(localStorage.getItem(LEGACY_MIGRATION_MARKER_KEY)).toBeNull();
    await importLegacyHistory();

    expect(localStorage.getItem(LEGACY_STORAGE_KEY)).toBe(JSON.stringify([RECORD]));
    expect(localStorage.getItem(LEGACY_MIGRATION_MARKER_KEY)).toContain('importedCount');
  });

  it('imports legacy arrays in original order and does not duplicate on repeated import', async () => {
    const db = { ...createMemoryStorageDb(), close: () => undefined };
    resetWithDb(db);
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD, SECOND_RECORD]));

    await importLegacyHistory();
    __resetStorageForTests({ dbFactory: () => db });
    await expect(importLegacyHistory()).resolves.toBe(0);

    expect((await getHistory()).map((record) => record.id)).toEqual(['analysis-id', 'second-id']);
  });

  it('shares concurrent initialization without implicitly importing legacy records', async () => {
    const db = createMemoryStorageDb();
    const importSpy = vi.spyOn(db, 'importMissingRecords');
    resetWithDb(db);
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD, SECOND_RECORD]));

    const [first, second, third] = await Promise.all([getHistory(), initializeStorage(), getHistory()]);

    expect(first).toEqual([]);
    expect(second.ok).toBe(true);
    expect(third).toEqual([]);
    expect(importSpy).not.toHaveBeenCalled();
  });

  it('explicit import merges missing legacy ids without overwriting account records', async () => {
    const newerA = { ...RECORD, imageName: 'indexeddb-a.png' };
    resetWithDb(createMemoryStorageDb([newerA]));
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD, SECOND_RECORD]));

    await expect(importLegacyHistory()).resolves.toBe(1);
    const history = await getHistory();

    expect(history.map((record) => record.id)).toEqual(['analysis-id', 'second-id']);
    expect(history[0].imageName).toBe('indexeddb-a.png');
  });

  it('saveAnalysis prepends normalized new records and replaces duplicate ids in place', async () => {
    await saveAnalysis(RECORD);
    await saveAnalysis(SECOND_RECORD);
    await saveAnalysis({ ...RECORD, imageName: 'updated.png', annotationStatus: { 0: 'validated', 9: 'validated' } });

    const history = await getHistory();

    expect(history.map((record) => record.id)).toEqual(['second-id', 'analysis-id']);
    expect(history[1].imageName).toBe('updated.png');
    expect(history[1].annotationStatus).toEqual({ 0: 'validated' });
  });

  it('getAnalysisById returns matching normalized record or null', async () => {
    await saveAnalysis(RECORD);

    await expect(getAnalysisById('analysis-id')).resolves.toEqual(expect.objectContaining({ id: 'analysis-id', annotationStatus: {} }));
    await expect(getAnalysisById('missing')).resolves.toBeNull();
  });

  it('updateElements updates elements, num_elements, and normalized status', async () => {
    await saveAnalysis({ ...RECORD, annotationStatus: { 0: 'validated', 1: 'draft' } });

    const nextElements = [{ ...RECORD.result.elements[0], class_name: 'renamed' }];
    await expect(updateElements('analysis-id', nextElements)).resolves.toBe(true);

    const [stored] = await getHistory();
    expect(stored.result.elements).toHaveLength(1);
    expect(stored.result.num_elements).toBe(1);
    expect(stored.annotationStatus).toEqual({ 0: 'validated' });
  });

  it('updateElements accepts explicit annotation status updates and returns false for missing ids', async () => {
    await saveAnalysis(RECORD);

    await expect(updateElements('analysis-id', RECORD.result.elements, { 0: 'validated' })).resolves.toBe(true);
    await expect(updateElements('missing', RECORD.result.elements)).resolves.toBe(false);

    expect((await getHistory())[0].annotationStatus).toEqual({ 0: 'validated' });
  });

  it('deleteAnalysis removes only requested ids and is idempotent', async () => {
    await saveAnalysis(RECORD);
    await saveAnalysis(SECOND_RECORD);

    await deleteAnalysis('second-id');
    await deleteAnalysis('missing');

    expect((await getHistory()).map((record) => record.id)).toEqual(['analysis-id']);
  });

  it('does not expose unowned legacy history when account storage fails', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));
    __resetStorageForTests({
      dbFactory: () => {
        throw new Error('open failed');
      },
    });

    await expect(getHistory()).rejects.toThrow('open failed');
    await expect(saveAnalysis(cloneRecord(RECORD))).rejects.toThrow('open failed');
    expect(getLastStorageInitResult()).toEqual(expect.objectContaining({ ok: false }));
  });

  it('does not call a storage failure a missing analysis', async () => {
    __resetStorageForTests({ dbFactory: () => { throw new Error('open failed'); } });
    await expect(getAnalysisById('analysis-id')).rejects.toThrow('open failed');
  });

  it('reopens the database after a transient history read failure', async () => {
    const stored = createMemoryStorageDb([RECORD]);
    let openings = 0;
    let reads = 0;
    __resetStorageForTests({ dbFactory: () => {
      openings += 1;
      return {
        ...stored,
        listRecords: async () => {
          reads += 1;
          if (reads === 2) throw new Error('read failed');
          return stored.listRecords();
        },
        close: () => undefined,
      };
    } });
    await expect(getHistory()).rejects.toThrow('read failed');
    await expect(getHistory()).resolves.toHaveLength(1);
    expect(openings).toBe(2);
  });

  it('keeps records isolated across accounts', async () => {
    const databases = new Map<string, AnalysisStorageDb>();
    __resetStorageForTests({
      dbFactory: (name) => {
        if (!databases.has(name)) databases.set(name, { ...createMemoryStorageDb(), close: () => undefined });
        return databases.get(name)!;
      },
    });
    setStorageAccount('alice');
    await saveAnalysis(RECORD);
    setStorageAccount('bob');
    await expect(getHistory()).resolves.toEqual([]);
    await saveAnalysis(SECOND_RECORD);
    setStorageAccount('alice');
    expect((await getHistory()).map((record) => record.id)).toEqual(['analysis-id']);
    setStorageAccount('bob');
    expect((await getHistory()).map((record) => record.id)).toEqual(['second-id']);
  });

  it('reserves unowned history recovery for one explicitly allowed account', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));
    setStorageAccount('alice');
    await expect(getLegacyImportCount()).resolves.toBe(0);
    await expect(importLegacyHistory()).rejects.toThrow('administration');
    setStorageAccount('alice', true);
    await expect(importLegacyHistory()).resolves.toBe(1);
    setStorageAccount('bob', true);
    await expect(getLegacyImportCount()).resolves.toBe(0);
    await expect(importLegacyHistory()).rejects.toThrow('autre compte');
  });

  it('does not assign legacy history to a failed importer and hides it after a successful import', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));
    let failImport = true;
    const databases = new Map<string, AnalysisStorageDb>();
    __resetStorageForTests({ dbFactory: (name) => {
      if (!databases.has(name)) {
        const stored = createMemoryStorageDb();
        databases.set(name, {
          ...stored,
          close: () => undefined,
          importMissingRecords: async (records) => {
            if (failImport) throw new Error('import failed');
            return stored.importMissingRecords(records);
          },
        });
      }
      return databases.get(name)!;
    } });
    setStorageAccount('alice', true);
    await expect(importLegacyHistory()).rejects.toThrow('import failed');
    expect(localStorage.getItem('clinic-codex-legacy-owner')).toBeNull();

    failImport = false;
    setStorageAccount('bob', true);
    await expect(importLegacyHistory()).resolves.toBe(1);
    await expect(getLegacyImportCount()).resolves.toBe(0);
    expect((await getHistory()).map((record) => record.id)).toEqual(['analysis-id']);
  });

  it('does not complete legacy import when the old IndexedDB cannot be read', async () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify([RECORD]));
    vi.stubGlobal('indexedDB', { open: () => { throw new Error('old database blocked'); } });
    try {
      await expect(getLegacyImportCount()).rejects.toThrow('old database blocked');
      await expect(importLegacyHistory()).rejects.toThrow('old database blocked');
      expect(localStorage.getItem(LEGACY_MIGRATION_MARKER_KEY)).toBeNull();
      expect(localStorage.getItem('clinic-codex-legacy-owner')).toBeNull();
      await expect(getHistory()).resolves.toEqual([]);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
