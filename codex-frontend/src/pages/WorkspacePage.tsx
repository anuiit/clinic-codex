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
  PanelLeftClose,
  PanelLeftOpen,
  ZoomIn,
  ZoomOut,
  Maximize2,
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
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [historyOpen, setHistoryOpen] = useState(() => !resolveCurrentRecord(getHistory(), initialPreferredId));
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
    if (zoom !== 1) {
      setZoom(1);
    }
    if (panOffset.x !== 0 || panOffset.y !== 0) {
      setPanOffset({ x: 0, y: 0 });
    }
    setOverlayMode('all');
  }, [panOffset.x, panOffset.y, zoom]);

  const selectRecord = useCallback((record: AnalysisRecord | null) => {
    resetInspectionState();
    setCurrentRecord(record);
    const nextHistoryOpen = !record;
    if (historyOpen !== nextHistoryOpen) {
      setHistoryOpen(nextHistoryOpen);
    }
  }, [historyOpen, resetInspectionState]);

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

  const getElementIndexFromCanvasPoint = useCallback((offsetX: number, offsetY: number) => {
    if (!currentRecord || !imageRef.current) {
      return null;
    }

    const { width: renderedWidth, height: renderedHeight } = imageRef.current.getBoundingClientRect();
    const [origW, origH] = currentRecord.result.image_size;

    if (!renderedWidth || !renderedHeight || !origW || !origH) {
      return null;
    }

    const imageX = offsetX / (renderedWidth / origW);
    const imageY = offsetY / (renderedHeight / origH);
    let matchedIdx: number | null = null;

    currentRecord.result.elements.forEach((element, idx) => {
      const [x, y, w, h] = element.bbox;
      if (imageX >= x && imageX <= x + w && imageY >= y && imageY <= y + h) {
        matchedIdx = idx;
      }
    });

    return matchedIdx;
  }, [currentRecord]);

  const handleEditorHandoff = () => {
    if (!currentRecord) {
      return;
    }

    navigate(
      focusedIdx !== null
        ? `/annotate/${currentRecord.id}?element=${focusedIdx}`
        : `/annotate/${currentRecord.id}`,
    );
  };

  return (
    <div className="space-y-6">
      <section className="flex flex-col gap-3 rounded-2xl border border-stone-800 bg-stone-900/80 p-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3 px-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500 text-stone-950">
            <ImagePlus size={18} />
          </div>
          <span className="font-semibold tracking-tight text-stone-100">Codex Glyph Analyzer</span>
        </div>

        <div className="flex flex-1 items-center justify-end gap-3">
          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-400/10 px-3 py-1.5 text-xs text-red-300">
              <AlertCircle size={14} className="shrink-0" />
              <span className="max-w-[300px] truncate">{error}</span>
            </div>
          )}

          <div
            className={`flex items-center gap-3 rounded-xl border border-dashed px-4 py-2 transition-colors ${dragging ? 'border-amber-400 bg-amber-400/10' : 'border-stone-700/80 bg-stone-950/60 hover:border-stone-500'}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
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
              <div className="flex items-center gap-3">
                <img src={preview} alt="Preview" className="h-8 w-8 rounded object-cover" />
                <div className="flex flex-col">
                  <span className="max-w-[120px] truncate text-xs font-medium text-stone-100">{file?.name}</span>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    analyze();
                  }}
                  disabled={loading}
                  className="ml-2 inline-flex h-8 items-center justify-center gap-2 rounded-lg bg-amber-500 px-3 text-xs font-semibold text-stone-950 transition-colors hover:bg-amber-400 disabled:opacity-50"
                >
                  {loading ? <><Loader2 size={14} className="animate-spin" /> Analyzing…</> : <>Analyze</>}
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-sm text-stone-400">
                <Upload size={16} />
                <span>Drop a glyph image here (PNG, JPG, BMP) or click to browse</span>
              </div>
            )}
          </div>
        </div>
      </section>

      <div className={`grid gap-6 transition-all duration-300 ${historyOpen ? 'xl:grid-cols-[340px_minmax(0,1fr)]' : 'xl:grid-cols-[64px_minmax(0,1fr)]'}`}>
        <aside className={`flex flex-col transition-all duration-300 ${historyOpen ? 'rounded-[28px] border border-stone-800 bg-stone-900/80 p-4 shadow-[0_20px_60px_rgba(0,0,0,0.18)] sm:p-5' : 'items-center rounded-[24px] border border-stone-800 bg-stone-900/80 py-4 shadow-sm'}`}>
              {!historyOpen ? (
                <>
                  <div className="relative flex flex-col items-center">
                    <button
                      type="button"
                      onClick={() => setHistoryOpen(true)}
                      className="flex h-10 w-10 items-center justify-center rounded-xl bg-stone-950 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100"
                      title="Expand history"
                    >
                      <PanelLeftOpen size={18} />
                    </button>

                    {/* subtle count badge for collapsed rail */}
                    <div className="mt-2">
                      <span className="inline-flex items-center justify-center rounded-full bg-amber-500 px-2 py-0.5 text-[10px] font-semibold text-stone-950">{records.length}</span>
                    </div>
                  </div>

                  <div className="my-4 h-px w-6 bg-stone-800" />
                  <div className="text-[10px] uppercase tracking-widest text-stone-500" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>
                    History
                  </div>

                  {currentRecord && (
                    <div className="mt-4 flex flex-1 flex-col items-center gap-2">
                      <div className="h-px w-6 bg-stone-800" />
                      <img src={currentRecord.imageDataUrl} alt="Current" className="mt-2 h-8 w-8 rounded-lg border border-amber-500/50 object-cover opacity-80 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]" title={currentRecord.imageName} />
                    </div>
                  )}
                </>
              ) : (
            <>
              <div className="mb-4 flex items-start justify-between gap-3 border-b border-stone-800 pb-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.24em] text-stone-500">Analysis history</p>
                  <h2 className="mt-1 text-lg font-semibold text-stone-100">Saved runs</h2>
                </div>
                <div className="flex items-center gap-2">
                  <div className="rounded-xl border border-stone-800 bg-stone-950/70 px-3 py-1.5 text-right">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-stone-500">Items</div>
                    <div className="text-base font-semibold leading-tight text-stone-100">{filteredRecords.length}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setHistoryOpen(false)}
                    className="flex h-10 w-10 items-center justify-center rounded-xl bg-stone-950 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100"
                    title="Collapse history"
                  >
                    <PanelLeftClose size={18} />
                  </button>
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

              <div className="space-y-3 overflow-y-auto pr-1 xl:max-h-[calc(100vh-14rem)]">
                {records.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-stone-700 bg-stone-950/70 px-4 py-8 text-center text-sm text-stone-500">
                    No analyses yet.
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
            </>
          )}
        </aside>

        <section className="space-y-6">
          {currentRecord ? (
            <>
              <div className="grid gap-6 2xl:grid-cols-[minmax(0,1.04fr)_minmax(380px,0.96fr)]">
                <section className="flex flex-col min-h-[420px] rounded-[28px] border border-stone-800 bg-stone-900/80 p-5 lg:p-6">
                  <div className="flex flex-col gap-4 border-b border-stone-800 pb-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-3">
                      <h2 className="text-lg font-semibold text-stone-100 truncate max-w-[200px] sm:max-w-[300px]" title={currentRecord.imageName}>
                        {currentRecord.imageName}
                      </h2>
                      {focusedIdx !== null && (
                        <span className="rounded-md bg-amber-500/10 px-2 py-1 text-xs font-medium text-amber-500 border border-amber-500/20">
                          Focus: {focusedIdx}
                        </span>
                      )}
                    </div>
                    <div className="inline-flex rounded-lg border border-stone-700 bg-stone-950 p-1 shrink-0">
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

                  <div 
                    className="relative mt-5 flex flex-1 min-h-[360px] items-center justify-center overflow-hidden rounded-[24px] border border-stone-800 bg-stone-950"
                    onWheel={(e) => { 
                      e.preventDefault(); 
                      const zoomChange = e.deltaY < 0 ? 0.15 : -0.15;
                      setZoom(z => Math.max(0.25, Math.min(4, z + zoomChange)));
                    }}
                  >
                    <div style={{ transform: `scale(${zoom})`, transformOrigin: 'center center', transition: 'transform 0.1s ease' }} className="relative inline-block">
                      <img
                        ref={imageRef}
                        src={currentRecord.imageDataUrl}
                        alt={currentRecord.imageName}
                        className="block max-h-[72vh] max-w-full rounded-lg object-contain"
                      />
                      <canvas
                        ref={canvasRef}
                        className="absolute left-0 top-0 cursor-pointer"
                        onClick={(event) => {
                          if (!currentRecord) {
                            return;
                          }

                          const matchedIdx = getElementIndexFromCanvasPoint(event.nativeEvent.offsetX, event.nativeEvent.offsetY);
                          if (matchedIdx === null) {
                            return;
                          }

                          setFocusedIdx(matchedIdx);
                          toggleExpand(matchedIdx);
                        }}
                        onMouseMove={(event) => {
                          const matchedIdx = getElementIndexFromCanvasPoint(event.nativeEvent.offsetX, event.nativeEvent.offsetY);
                          setHoveredIdx(matchedIdx);
                        }}
                        onMouseLeave={() => setHoveredIdx(null)}
                      />
                    </div>
                    
                    <div className="absolute bottom-4 right-4 flex items-center gap-1 rounded-xl border border-stone-700 bg-stone-900/90 p-1 backdrop-blur-sm">
                      <button type="button" onClick={() => setZoom(z => Math.min(4, z + 0.25))} className="rounded-lg p-2 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100" title="Zoom in">
                        <ZoomIn size={16} />
                      </button>
                      <button type="button" onClick={() => { setZoom(1); setPanOffset({x:0, y:0}); }} className="rounded-lg p-2 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100" title="Fit to view">
                        <Maximize2 size={16} />
                      </button>
                      <button type="button" onClick={() => setZoom(z => Math.max(0.25, z - 0.25))} className="rounded-lg p-2 text-stone-400 transition-colors hover:bg-stone-800 hover:text-stone-100" title="Zoom out">
                        <ZoomOut size={16} />
                      </button>
                    </div>
                  </div>
                </section>

                <section className="flex h-[72vh] min-h-[520px] flex-col overflow-hidden rounded-[28px] border border-stone-800 bg-stone-900/80 p-5 lg:p-6">
                  <div className="mb-4 flex items-start justify-between gap-4 border-b border-stone-800 pb-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.24em] text-stone-500">Proposal panel</p>
                      <h2 className="mt-1 text-lg font-semibold text-stone-100">Detected elements</h2>
                    </div>
                    <div className="text-right">
                      <button
                        type="button"
                        onClick={handleEditorHandoff}
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
                        <button type="button" onClick={handleEditorHandoff} className="mt-1 text-xs font-medium text-amber-400 hover:text-amber-300">
                          Go to annotation →
                        </button>
                      </div>
                    </div>
                  )}

                  <div
                    className="flex-1 space-y-1 overflow-y-auto pr-2"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (!currentRecord) return;
                      const total = currentRecord.result.elements.length;
                      if (total === 0) return;
                      if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                        e.preventDefault();
                        setFocusedIdx((prev) => (prev === null ? 0 : (prev + 1) % total));
                      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                        e.preventDefault();
                        setFocusedIdx((prev) => (prev === null ? total - 1 : (prev - 1 + total) % total));
                      } else if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        if (focusedIdx !== null) toggleExpand(focusedIdx);
                      } else if (e.key === 'Escape') {
                        setFocusedIdx(null);
                      }
                    }}
                  >
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
                        let destinationWidth = 48;
                        let destinationHeight = 48;
                        if (boxWidth >= boxHeight) {
                          destinationHeight = Math.max(1, 48 * (boxHeight / boxWidth));
                        } else {
                          destinationWidth = Math.max(1, 48 * (boxWidth / boxHeight));
                        }

                        return (
                          <div key={idx} className="flex flex-col">
                            <div
                              className={`flex items-center gap-2 px-3 py-2 rounded-xl border cursor-pointer transition-colors ${
                                isFocused ? 'border-amber-500/60 bg-amber-500/5' : isHovered ? 'border-stone-700 bg-stone-900' : 'border-stone-800 bg-stone-950'
                              }`}
                              onClick={() => {
                                setFocusedIdx(idx);
                                toggleExpand(idx);
                              }}
                              onMouseEnter={() => setHoveredIdx(idx)}
                              onMouseLeave={() => setHoveredIdx((current) => (current === idx ? null : current))}
                              tabIndex={0}
                            >
                              <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded text-xs font-bold ${indexBadgeClasses}`}>
                                {idx}
                              </span>
                              <span className="flex-1 truncate text-sm text-stone-100">{displayClass}</span>
                              {hasAnnotation && <CheckCircle2 size={12} className="text-green-400 shrink-0" />}
                              <span className={`shrink-0 rounded px-1.5 py-0.5 text-xs font-medium ${badgeClasses}`}>
                                {(element.confidence * 100).toFixed(1)}%
                              </span>
                              {isExpanded ? (
                                <ChevronUp size={14} className="shrink-0 text-stone-500" />
                              ) : (
                                <ChevronDown size={14} className="shrink-0 text-stone-500" />
                              )}
                            </div>

                            <div className={isExpanded ? "mt-1 rounded-xl border border-amber-500/30 bg-stone-950/80 p-3" : "hidden"}>
                              <div className="flex gap-4">
                                <div className="flex h-[48px] w-[48px] shrink-0 items-center justify-center overflow-hidden rounded border border-stone-800/50 bg-stone-900">
                                  <canvas
                                    ref={(canvas) => {
                                      cropCanvasRefs.current[idx] = canvas;
                                    }}
                                    width={destinationWidth}
                                    height={destinationHeight}
                                    className="block rounded bg-stone-800"
                                  />
                                </div>

                                <div className="flex-1">
                                  <div className="mb-1 text-xs font-medium text-stone-400">Alternative predictions:</div>
                                  <div className="space-y-1">
                                    {element.top_k.map((topItem, topItemIndex) => (
                                      <div key={topItemIndex} className="flex items-center justify-between text-xs">
                                        <span className="text-stone-300">{topItem.class_name}</span>
                                        <span className="text-stone-500">{(topItem.confidence * 100).toFixed(1)}%</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>

                              {element.rejected && (
                                <div className="mt-2 flex items-start gap-2 rounded-lg border border-red-400/20 bg-red-400/10 p-2 text-red-400">
                                  <AlertCircle size={14} className="mt-0.5 shrink-0" />
                                  <span className="text-xs">Low confidence prediction. Review recommended.</span>
                                </div>
                              )}

                              {(isFocused || element.rejected) && (
                                <div className="mt-3 flex justify-end">
                                  <button
                                    type="button"
                                    onClick={handleEditorHandoff}
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
              <h2 className="mt-5 text-xl font-semibold text-stone-100">No analysis selected</h2>
              <p className="mt-2 text-sm text-stone-400">
                Upload a new glyph or select a saved run from history.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
