import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent } from 'react';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Edit3,
  ImagePlus,
  Info,
  Loader2,
  Search,
  Trash2,
  Upload,
} from 'lucide-react';
import { segmentGlyph } from '../services/api';
import { deleteAnalysis, getHistory, saveAnalysis } from '../services/storage';
import type { AnalysisRecord, DetectedElement } from '../types';

type OverlayMode = 'all' | 'focused' | 'hidden';

function resolveCurrentRecord(records: AnalysisRecord[], preferredId?: string | null) {
  if (preferredId) {
    return records.find((record) => record.id === preferredId) ?? records[0] ?? null;
  }

  return records[0] ?? null;
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  elements: DetectedElement[],
  imageSize: [number, number],
  canvasSize: { width: number; height: number },
  hoveredIdx: number | null,
  focusedIdx: number | null,
  overlayMode: OverlayMode,
) {
  const { width, height } = canvasSize;
  ctx.clearRect(0, 0, width, height);

  if (overlayMode === 'hidden') {
    return;
  }

  const [origW, origH] = imageSize;
  const scaleX = width / origW;
  const scaleY = height / origH;

  elements.forEach((el, idx) => {
    if (overlayMode === 'focused' && focusedIdx !== null && focusedIdx !== idx) {
      return;
    }

    const [x, y, w, h] = el.bbox;
    const scaledX = x * scaleX;
    const scaledY = y * scaleY;
    const scaledW = w * scaleX;
    const scaledH = h * scaleY;

    let strokeColor = el.rejected ? '#ef4444' : '#f59e0b';
    let lineWidth = 2;
    let badgeColor = el.rejected ? '#ef4444' : '#f59e0b';
    const textColor = '#0c0a09';

    if (idx === focusedIdx) {
      strokeColor = '#ffffff';
      lineWidth = 3;
      badgeColor = '#ffffff';
    } else if (idx === hoveredIdx) {
      strokeColor = '#fbbf24';
      lineWidth = 2.5;
      badgeColor = '#fbbf24';
    }

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

    ctx.fillStyle = badgeColor;
    ctx.fillRect(scaledX, Math.max(0, scaledY - 24), 28, 24);

    ctx.fillStyle = textColor;
    ctx.font = 'bold 14px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${idx}`, scaledX + 14, Math.max(0, scaledY - 24) + 12);
  });
}

export default function WorkspacePage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const initialPreferredId = searchParams.get('analysis');

  const [records, setRecords] = useState<AnalysisRecord[]>(() => getHistory());
  const [currentRecord, setCurrentRecord] = useState<AnalysisRecord | null>(() => resolveCurrentRecord(getHistory(), initialPreferredId));
  const [filter, setFilter] = useState('');

  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [expandedItems, setExpandedItems] = useState<Record<number, boolean>>({});
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [focusedIdx, setFocusedIdx] = useState<number | null>(null);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('all');

  const inputRef = useRef<HTMLInputElement>(null);
  const currentFileRef = useRef<File | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const cropCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);

  const resetInspectionState = useCallback(() => {
    cropCanvasRefs.current = [];
    setExpandedItems({});
    setHoveredIdx(null);
    setFocusedIdx(null);
    setOverlayMode('all');
  }, []);

  const selectRecord = useCallback((record: AnalysisRecord | null) => {
    resetInspectionState();
    setCurrentRecord(record);
  }, [resetInspectionState]);

  const syncRecords = useCallback((preferredId?: string | null) => {
    const nextRecords = getHistory();
    setRecords(nextRecords);
    selectRecord(resolveCurrentRecord(nextRecords, preferredId));
  }, [selectRecord]);

  useEffect(() => {
    if (location.pathname === '/' && searchParams.get('analysis')) {
      navigate('/', { replace: true });
    }
  }, [location.pathname, navigate, searchParams]);

  useEffect(() => {
    if (!currentRecord || !canvasRef.current || !imageRef.current) {
      return;
    }

    const image = imageRef.current;

    const drawAllCanvases = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return;
      }

      const { width, height } = image.getBoundingClientRect();
      canvas.width = width;
      canvas.height = height;

      drawOverlay(
        ctx,
        currentRecord.result.elements,
        currentRecord.result.image_size,
        { width, height },
        hoveredIdx,
        focusedIdx,
        overlayMode,
      );

      currentRecord.result.elements.forEach((element, idx) => {
        const cropCanvas = cropCanvasRefs.current[idx];
        if (!cropCanvas) {
          return;
        }

        const cropCtx = cropCanvas.getContext('2d');
        if (!cropCtx) {
          return;
        }

        const [x, y, w, h] = element.bbox;
        cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
        cropCtx.drawImage(image, x, y, w, h, 0, 0, cropCanvas.width, cropCanvas.height);
      });
    };

    if (image.complete) {
      drawAllCanvases();
    }

    image.addEventListener('load', drawAllCanvases);
    window.addEventListener('resize', drawAllCanvases);

    return () => {
      image.removeEventListener('load', drawAllCanvases);
      window.removeEventListener('resize', drawAllCanvases);
    };
  }, [currentRecord, focusedIdx, hoveredIdx, overlayMode]);

  const filteredRecords = useMemo(() => {
    if (!filter) {
      return records;
    }

    const query = filter.toLowerCase();
    return records.filter((record) => {
      const names = record.result.elements.map((element) => element.class_name.toLowerCase());
      const annotated = Object.values(record.annotations ?? {}).map((value) => value.toLowerCase());
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

    elements.forEach((element, idx) => {
      const finalClass = (currentRecord.annotations ?? {})[idx] ?? element.class_name;
      classCounts[finalClass] = (classCounts[finalClass] || 0) + 1;
    });

    let topClass = 'None';
    let maxCount = 0;
    Object.entries(classCounts).forEach(([className, count]) => {
      if (count > maxCount) {
        maxCount = count;
        topClass = className;
      }
    });

    return {
      total: elements.length,
      rejectedCount,
      topClass,
    };
  }, [currentRecord]);

  const handleFile = useCallback((nextFile: File) => {
    currentFileRef.current = nextFile;
    setFile(nextFile);
    setPreview(null);
    setError(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      if (currentFileRef.current === nextFile) {
        setPreview(event.target?.result as string);
      }
    };
    reader.readAsDataURL(nextFile);
  }, []);

  const onDrop = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const nextFile = event.dataTransfer.files[0];
    if (nextFile) {
      handleFile(nextFile);
    }
  }, [handleFile]);

  const analyze = async () => {
    if (!file || !preview) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await segmentGlyph(file);
      const record: AnalysisRecord = {
        id: crypto.randomUUID(),
        imageName: file.name,
        imageDataUrl: preview,
        timestamp: Date.now(),
        result,
        annotations: {},
      };

      saveAnalysis(record);
      syncRecords(record.id);
    } catch (issue) {
      setError(issue instanceof Error ? issue.message : 'API error — is the Flask server running on port 5000?');
    } finally {
      setLoading(false);
    }
  };

  const removeRecord = (id: string) => {
    deleteAnalysis(id);
    syncRecords(currentRecord?.id === id ? null : currentRecord?.id);
  };

  const toggleExpand = (idx: number) => {
    setExpandedItems((previous) => ({ ...previous, [idx]: !previous[idx] }));
  };

  const goToAnnotation = () => {
    if (currentRecord) {
      navigate(`/annotate/${currentRecord.id}`);
    }
  };

  return (
    <div className="space-y-6">
      <section className="overflow-hidden rounded-[28px] border border-amber-500/15 bg-[radial-gradient(circle_at_top_left,_rgba(245,158,11,0.22),_transparent_38%),linear-gradient(135deg,rgba(28,25,23,0.98),rgba(12,10,9,0.94))] shadow-[0_24px_80px_rgba(0,0,0,0.28)]">
        <div className="grid gap-6 px-5 py-5 lg:grid-cols-[1.05fr_0.95fr] lg:px-7 lg:py-7">
          <div className="space-y-5">
            <div className="space-y-3">
              <div className="inline-flex items-center gap-2 rounded-full border border-amber-400/20 bg-amber-400/10 px-3 py-1 text-xs uppercase tracking-[0.32em] text-amber-300">
                <ImagePlus size={14} /> Unified analysis workspace
              </div>
              <div>
                <h1 className="text-3xl font-semibold tracking-tight text-stone-50 sm:text-4xl">Upload, inspect, and reopen glyph analyses from one surface.</h1>
                <p className="mt-3 max-w-2xl text-sm leading-6 text-stone-300 sm:text-base">
                  Keep the upload bench, analysis history, and current inspection canvas visible together so the analyst never has to hop across routes.
                </p>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-2xl border border-stone-800/80 bg-stone-950/60 px-4 py-4">
                <div className="text-xs uppercase tracking-[0.24em] text-stone-500">Stored runs</div>
                <div className="mt-2 text-3xl font-semibold text-stone-100">{records.length}</div>
              </div>
              <div className="rounded-2xl border border-stone-800/80 bg-stone-950/60 px-4 py-4">
                <div className="text-xs uppercase tracking-[0.24em] text-stone-500">Current file</div>
                <div className="mt-2 truncate text-base font-medium text-stone-100">{file?.name ?? currentRecord?.imageName ?? 'Nothing loaded'}</div>
              </div>
              <div className="rounded-2xl border border-stone-800/80 bg-stone-950/60 px-4 py-4">
                <div className="text-xs uppercase tracking-[0.24em] text-stone-500">Selection</div>
                <div className="mt-2 text-base font-medium text-stone-100">{currentRecord ? `${currentRecord.result.num_elements} proposals ready` : 'Choose a record'}</div>
              </div>
            </div>

            {error && (
              <div className="flex items-start gap-2 rounded-2xl border border-red-400/20 bg-red-400/10 p-4 text-red-300">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                <span className="text-sm">{error}</span>
              </div>
            )}

            <div className="flex flex-col gap-3 sm:flex-row">
              <button
                type="button"
                onClick={analyze}
                disabled={!file || !preview || loading}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-5 py-3 text-sm font-semibold text-stone-950 transition-colors hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? <><Loader2 size={18} className="animate-spin" /> Analyzing…</> : <>Analyze glyph</>}
              </button>
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-stone-700 bg-stone-900/80 px-5 py-3 text-sm font-medium text-stone-200 transition-colors hover:border-stone-500 hover:text-stone-50"
              >
                <Upload size={16} /> Choose another image
              </button>
            </div>
          </div>

          <div
            className={`group relative overflow-hidden rounded-[24px] border-2 border-dashed p-4 transition-colors sm:p-5 ${dragging ? 'border-amber-400 bg-amber-400/10' : 'border-stone-700/80 bg-stone-950/60 hover:border-stone-500'}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/bmp,image/*"
              className="hidden"
              onChange={(event) => {
                const nextFile = event.target.files?.[0];
                if (nextFile) {
                  handleFile(nextFile);
                }
              }}
            />

            {preview ? (
              <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_220px] sm:items-end">
                <div className="overflow-hidden rounded-2xl border border-stone-800 bg-stone-950/80 p-2">
                  <img src={preview} alt="Preview" className="max-h-72 w-full rounded-xl object-contain" />
                </div>
                <div className="space-y-3 rounded-2xl border border-stone-800 bg-stone-950/80 p-4 text-sm text-stone-300">
                  <div>
                    <div className="text-xs uppercase tracking-[0.24em] text-stone-500">Queued upload</div>
                    <div className="mt-2 truncate font-medium text-stone-100">{file?.name ?? 'Untitled image'}</div>
                  </div>
                  {file && <div className="text-stone-400">{(file.size / 1024).toFixed(0)} KB ready for segmentation</div>}
                  <div className="rounded-xl border border-amber-400/15 bg-amber-400/10 px-3 py-2 text-xs text-amber-200">
                    Run analysis to save this image directly into the left-hand history column.
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex min-h-[260px] flex-col items-center justify-center gap-4 rounded-[20px] bg-stone-950/70 px-6 text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-amber-400/20 bg-amber-400/10 text-amber-300">
                  <Upload size={28} />
                </div>
                <div>
                  <p className="text-lg font-medium text-stone-100">Drop a codex glyph here</p>
                  <p className="mt-2 text-sm text-stone-400">PNG, JPG, BMP supported. Click to browse or drag directly into the upload bench.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="rounded-[28px] border border-stone-800 bg-stone-900/80 p-4 shadow-[0_20px_60px_rgba(0,0,0,0.18)] sm:p-5">
          <div className="mb-4 flex items-start justify-between gap-3 border-b border-stone-800 pb-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-stone-500">Analysis history</p>
              <h2 className="mt-1 text-lg font-semibold text-stone-100">Reopen saved runs</h2>
            </div>
            <div className="rounded-xl border border-stone-800 bg-stone-950/70 px-3 py-2 text-right">
              <div className="text-[11px] uppercase tracking-[0.18em] text-stone-500">Visible</div>
              <div className="text-lg font-semibold text-stone-100">{filteredRecords.length}</div>
            </div>
          </div>

          <div className="relative mb-4">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-500" />
            <input
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Filter by glyph or class"
              className="w-full rounded-xl border border-stone-700 bg-stone-950 px-10 py-2.5 text-sm text-stone-100 outline-none transition-colors focus:border-amber-400"
            />
          </div>

          <div className="space-y-3 overflow-y-auto pr-1 xl:max-h-[calc(100vh-18rem)]">
            {records.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-stone-700 bg-stone-950/70 px-4 py-8 text-center text-sm text-stone-500">
                No analyses yet. Upload a glyph above to seed this workspace.
              </div>
            ) : filteredRecords.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-stone-700 bg-stone-950/70 px-4 py-8 text-center text-sm text-stone-500">
                Nothing matches this filter.
              </div>
            ) : (
              filteredRecords.map((record) => {
                const isActive = currentRecord?.id === record.id;
                const badges = [
                  ...new Set(
                    record.result.elements
                      .map((element, idx) => (!element.rejected ? (record.annotations ?? {})[idx] ?? element.class_name : null))
                      .filter((value): value is string => Boolean(value)),
                  ),
                ].slice(0, 3);

                return (
                  <div
                    key={record.id}
                    role="button"
                    tabIndex={0}
                    onClick={() => selectRecord(record)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        selectRecord(record);
                      }
                    }}
                    className={`group rounded-2xl border p-3 transition-all ${isActive ? 'border-amber-500/60 bg-amber-500/10 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]' : 'border-stone-800 bg-stone-950/70 hover:border-stone-700'}`}
                  >
                    <div className="flex items-start gap-3">
                      <img src={record.imageDataUrl} alt={record.imageName} className="h-20 w-20 rounded-xl border border-stone-800 object-cover" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <p className="truncate text-sm font-medium text-stone-100">{record.imageName}</p>
                            <p className="mt-1 text-xs text-stone-500">{new Date(record.timestamp).toLocaleDateString()} · {record.result.num_elements} elements</p>
                          </div>
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              removeRecord(record.id);
                            }}
                            className="rounded-lg p-1.5 text-stone-500 transition-colors hover:bg-stone-800 hover:text-red-300"
                            aria-label={`Delete ${record.imageName}`}
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {badges.map((badge) => (
                            <span key={badge} className="rounded-md bg-amber-400/10 px-2 py-1 text-[11px] text-amber-300">
                              {badge}
                            </span>
                          ))}
                          {record.result.elements.some((element) => element.rejected) && (
                            <span className="rounded-md bg-red-400/10 px-2 py-1 text-[11px] text-red-300">
                              {record.result.elements.filter((element) => element.rejected).length} rejected
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        <section className="space-y-6">
          {currentRecord ? (
            <>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="rounded-2xl border border-stone-800 bg-stone-900/80 p-4">
                  <div className="text-xs uppercase tracking-[0.24em] text-stone-500">Total elements</div>
                  <div className="mt-2 text-3xl font-semibold text-stone-100">{stats?.total ?? 0}</div>
                </div>
                <div className="rounded-2xl border border-stone-800 bg-stone-900/80 p-4">
                  <div className="text-xs uppercase tracking-[0.24em] text-stone-500">Rejected</div>
                  <div className="mt-2 text-3xl font-semibold text-red-400">{stats?.rejectedCount ?? 0}</div>
                </div>
                <div className="rounded-2xl border border-stone-800 bg-stone-900/80 p-4">
                  <div className="text-xs uppercase tracking-[0.24em] text-stone-500">Top predicted</div>
                  <div className="mt-2 truncate text-2xl font-semibold text-amber-400">{stats?.topClass ?? 'None'}</div>
                </div>
              </div>

              <div className="grid gap-6 2xl:grid-cols-[minmax(0,1.04fr)_minmax(380px,0.96fr)]">
                <section className="min-h-[420px] rounded-[28px] border border-stone-800 bg-stone-900/80 p-5 lg:p-6">
                  <div className="flex flex-col gap-4 border-b border-stone-800 pb-4 sm:flex-row sm:items-start sm:justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.24em] text-stone-500">Inspection workspace</p>
                      <h2 className="mt-1 text-lg font-semibold text-stone-100">Image overlay</h2>
                      <p className="mt-1 text-sm text-stone-400">Review the detected regions against the original glyph before opening annotation.</p>
                    </div>
                    <div className="inline-flex rounded-lg border border-stone-700 bg-stone-950 p-1">
                      {(['all', 'focused', 'hidden'] as OverlayMode[]).map((mode) => {
                        const isActive = overlayMode === mode;
                        return (
                          <button
                            key={mode}
                            type="button"
                            onClick={() => setOverlayMode(mode)}
                            className={`rounded-md px-3 py-1.5 text-sm font-medium capitalize transition-colors ${isActive ? 'bg-amber-500 text-stone-950' : 'text-stone-400 hover:text-stone-100'}`}
                          >
                            {mode}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <div className="mt-5 grid gap-3 text-sm sm:grid-cols-2">
                    <div className="rounded-xl border border-stone-800 bg-stone-950/70 px-4 py-3">
                      <div className="text-stone-500">Image</div>
                      <div className="mt-1 truncate font-medium text-stone-100">{currentRecord.imageName}</div>
                    </div>
                    <div className="rounded-xl border border-stone-800 bg-stone-950/70 px-4 py-3">
                      <div className="text-stone-500">Overlay focus</div>
                      <div className="mt-1 font-medium text-stone-100">
                        {focusedIdx !== null
                          ? `${focusedIdx} (${(currentRecord.annotations ?? {})[focusedIdx] || currentRecord.result.elements[focusedIdx]?.class_name || 'Unknown'})`
                          : 'None selected'}
                      </div>
                    </div>
                  </div>

                  <div className="relative mt-5 flex min-h-[360px] items-center justify-center overflow-hidden rounded-[24px] border border-stone-800 bg-stone-950">
                    <div className="relative inline-block max-w-full p-3">
                      <img
                        ref={imageRef}
                        src={currentRecord.imageDataUrl}
                        alt={currentRecord.imageName}
                        className="block max-h-[72vh] max-w-full rounded-lg object-contain"
                      />
                      <canvas ref={canvasRef} className="pointer-events-none absolute left-3 top-3" />
                    </div>
                  </div>
                </section>

                <section className="flex h-[72vh] min-h-[520px] flex-col overflow-hidden rounded-[28px] border border-stone-800 bg-stone-900/80 p-5 lg:p-6">
                  <div className="mb-4 flex items-start justify-between gap-4 border-b border-stone-800 pb-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.24em] text-stone-500">Proposal panel</p>
                      <h2 className="mt-1 text-lg font-semibold text-stone-100">Detected elements</h2>
                      <p className="mt-1 text-sm text-stone-400">Browse every proposal, expand alternatives, and lock focus for the overlay.</p>
                    </div>
                    <div className="space-y-2 text-right">
                      <div className="rounded-xl border border-stone-800 bg-stone-950/70 px-3 py-2">
                        <div className="text-xs text-stone-500">Visible proposals</div>
                        <div className="text-lg font-semibold text-stone-100">{currentRecord.result.elements.length}</div>
                      </div>
                      <button
                        type="button"
                        onClick={goToAnnotation}
                        className="inline-flex items-center gap-2 rounded-xl bg-amber-500 px-4 py-2 text-sm font-semibold text-stone-950 transition-colors hover:bg-amber-400"
                      >
                        <Edit3 size={16} /> Annotate current record
                      </button>
                    </div>
                  </div>

                  {stats && stats.rejectedCount === stats.total && stats.total > 0 && (
                    <div className="mb-4 flex items-start gap-3 rounded-2xl border border-amber-500/20 bg-amber-500/10 p-3">
                      <Info className="mt-0.5 shrink-0 text-amber-500" size={18} />
                      <div>
                        <p className="text-sm text-stone-200">All proposals were rejected — consider annotating to improve results.</p>
                        <button type="button" onClick={goToAnnotation} className="mt-1 text-xs font-medium text-amber-400 hover:text-amber-300">
                          Go to annotation →
                        </button>
                      </div>
                    </div>
                  )}

                  <div className="flex-1 space-y-3 overflow-y-auto pr-2">
                    {currentRecord.result.elements.length === 0 ? (
                      <div className="flex h-full flex-col items-center justify-center rounded-2xl border border-dashed border-stone-800 bg-stone-950/50 p-6 text-center">
                        <Info className="mb-3 text-stone-600" size={32} />
                        <p className="text-stone-400">No elements detected — try a different image or check segmentation settings.</p>
                      </div>
                    ) : (
                      currentRecord.result.elements.map((element, idx) => {
                        const hasAnnotation = (currentRecord.annotations ?? {})[idx] !== undefined;
                        const displayClass = hasAnnotation ? (currentRecord.annotations ?? {})[idx] : element.class_name;
                        const isExpanded = expandedItems[idx];
                        const isFocused = focusedIdx === idx;
                        const isHovered = hoveredIdx === idx;
                        const badgeClasses = element.rejected
                          ? 'bg-red-400/10 text-red-400 border border-red-400/20'
                          : 'bg-amber-400/10 text-amber-400 border border-amber-400/20';
                        const indexBadgeClasses = element.rejected ? 'bg-red-500 text-stone-950' : 'bg-amber-500 text-stone-950';

                        const [, , boxWidth, boxHeight] = element.bbox;
                        let destinationWidth = 80;
                        let destinationHeight = 80;
                        if (boxWidth >= boxHeight) {
                          destinationHeight = Math.max(1, 80 * (boxHeight / boxWidth));
                        } else {
                          destinationWidth = Math.max(1, 80 * (boxWidth / boxHeight));
                        }

                        return (
                          <div
                            key={idx}
                            className={`overflow-hidden rounded-2xl border transition-colors ${isFocused ? 'border-amber-500/60 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]' : isHovered ? 'border-stone-700' : 'border-stone-800'} bg-stone-950`}
                            onMouseEnter={() => setHoveredIdx(idx)}
                            onMouseLeave={() => setHoveredIdx((current) => (current === idx ? null : current))}
                          >
                            <div
                              className={`flex cursor-pointer items-center justify-between p-3 transition-colors ${isFocused ? 'bg-amber-500/5' : 'hover:bg-stone-800/50'}`}
                              onClick={() => {
                                setFocusedIdx(idx);
                                toggleExpand(idx);
                              }}
                              onFocus={() => setFocusedIdx(idx)}
                              onBlur={() => setFocusedIdx((current) => (current === idx ? null : current))}
                              tabIndex={0}
                              role="button"
                            >
                              <div className="flex items-center gap-3">
                                <div className="flex h-[80px] w-[80px] shrink-0 items-center justify-center overflow-hidden rounded-lg border border-stone-800/50 bg-stone-900">
                                  <canvas
                                    ref={(canvas) => {
                                      cropCanvasRefs.current[idx] = canvas;
                                    }}
                                    width={destinationWidth}
                                    height={destinationHeight}
                                    className="block rounded bg-stone-800"
                                  />
                                </div>
                                <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-sm font-bold ${indexBadgeClasses}`}>
                                  {idx}
                                </span>
                                <div className="ml-1 flex flex-col">
                                  <span className="font-semibold text-stone-100">{displayClass}</span>
                                  {hasAnnotation && (
                                    <span className="mt-0.5 flex items-center gap-1 text-xs text-green-400">
                                      <CheckCircle2 size={12} /> Corrected
                                    </span>
                                  )}
                                </div>
                              </div>
                              <div className="ml-2 flex items-center gap-3">
                                {isFocused && <span className="hidden text-xs font-medium uppercase tracking-wide text-amber-400 sm:inline-block">Focused</span>}
                                <span className={`shrink-0 rounded-md px-2 py-1 text-sm font-medium ${badgeClasses}`}>
                                  {(element.confidence * 100).toFixed(1)}%
                                </span>
                                {isExpanded ? <ChevronUp size={18} className="shrink-0 text-stone-500" /> : <ChevronDown size={18} className="shrink-0 text-stone-500" />}
                              </div>
                            </div>

                            {isExpanded && (
                              <div className="border-t border-stone-800/50 bg-stone-950/50 p-3 pt-0">
                                <div className="mb-2 mt-3 text-sm font-medium text-stone-400">Alternative predictions:</div>
                                <div className="space-y-2">
                                  {element.top_k.map((topItem, topItemIndex) => (
                                    <div key={topItemIndex} className="flex items-center justify-between text-sm">
                                      <span className="text-stone-300">{topItem.class_name}</span>
                                      <span className="text-stone-500">{(topItem.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                  ))}
                                </div>

                                {element.rejected && (
                                  <div className="mt-3 flex items-start gap-2 rounded-lg border border-red-400/20 bg-red-400/10 p-2 text-red-400">
                                    <AlertCircle size={14} className="mt-0.5 shrink-0" />
                                    <span className="text-xs">Low confidence prediction. Review recommended.</span>
                                  </div>
                                )}

                                {(isFocused || element.rejected) && (
                                  <div className="mt-4 flex justify-end">
                                    <button
                                      type="button"
                                      onClick={goToAnnotation}
                                      className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition-colors ${
                                        element.rejected ? 'bg-red-500 text-stone-950 hover:bg-red-400' : 'bg-amber-500 text-stone-950 hover:bg-amber-400'
                                      }`}
                                    >
                                      <Edit3 size={14} />
                                      {element.rejected ? 'Correct this element' : 'Annotate'}
                                    </button>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })
                    )}
                  </div>
                </section>
              </div>
            </>
          ) : (
            <div className="flex min-h-[520px] flex-col items-center justify-center rounded-[28px] border border-dashed border-stone-700 bg-stone-900/60 px-6 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-amber-400/20 bg-amber-400/10 text-amber-300">
                <Info size={28} />
              </div>
              <h2 className="mt-5 text-2xl font-semibold text-stone-100">No analysis selected</h2>
              <p className="mt-3 max-w-xl text-sm leading-6 text-stone-400">
                Upload a new glyph or pick a saved run from history to open the overlay canvas, proposal cards, and annotation handoff controls.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
