import { useEffect, useRef, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Edit3, ChevronDown, ChevronUp, AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { getAnalysisById } from '../services/storage';
import type { DetectedElement } from '../types';

type OverlayMode = 'all' | 'focused' | 'hidden';

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

  if (overlayMode === 'hidden') return;

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

export default function DetailPage() {
  const { id } = useParams<{ id: string }>();
  const [expandedItems, setExpandedItems] = useState<Record<number, boolean>>({});
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [focusedIdx, setFocusedIdx] = useState<number | null>(null);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('all');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const cropCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const record = useMemo(() => (id ? getAnalysisById(id) : null), [id]);

  useEffect(() => {
    if (!record || !canvasRef.current || !imageRef.current) return;

    const image = imageRef.current;

    const drawAllCanvases = () => {
      const canvas = canvasRef.current;
      if (!canvas || !image) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const { width, height } = image.getBoundingClientRect();
      canvas.width = width;
      canvas.height = height;

      drawOverlay(
        ctx,
        record.result.elements,
        record.result.image_size,
        { width, height },
        hoveredIdx,
        focusedIdx,
        overlayMode,
      );

      record.result.elements.forEach((el, idx) => {
        const cropCanvas = cropCanvasRefs.current[idx];
        if (!cropCanvas) return;
        const cropCtx = cropCanvas.getContext('2d');
        if (!cropCtx) return;

        const [x, y, w, h] = el.bbox;
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
  }, [record, hoveredIdx, focusedIdx, overlayMode]);

  const stats = useMemo(() => {
    if (!record) return null;
    const elements = record.result.elements;
    const rejectedCount = elements.filter(e => e.rejected).length;
    
    const classCounts: Record<string, number> = {};
    elements.forEach((e, i) => {
      const finalClass = (record.annotations ?? {})[i] ?? e.class_name;
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

    return { total: elements.length, rejectedCount, topClass };
  }, [record]);

  const toggleExpand = (idx: number) => {
    setExpandedItems(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  if (!record) {
    return (
      <div className="max-w-2xl mx-auto text-center py-12">
        <h2 className="text-2xl font-bold text-stone-100 mb-4">Analysis not found</h2>
        <Link to="/dashboard" className="text-amber-400 hover:text-amber-300 inline-flex items-center gap-2">
          <ArrowLeft size={18} /> Back to History
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/dashboard" className="text-stone-400 hover:text-stone-100 flex items-center gap-2 transition-colors">
            <ArrowLeft size={18} /> Back to history
          </Link>
          <h1 className="text-2xl font-bold">Analysis Details</h1>
        </div>
        <Link
          to={`/annotate/${record.id}`}
          className="bg-amber-500 hover:bg-amber-400 text-stone-950 font-semibold py-2 px-4 rounded-lg flex items-center gap-2 transition-colors"
        >
          <Edit3 size={18} /> Annotate rejected
        </Link>
      </div>

      {stats && (
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-stone-900 border border-stone-800 rounded-xl p-4 flex flex-col">
            <span className="text-stone-400 text-sm">Total Elements</span>
            {stats.total === 0 ? (
              <span className="text-lg font-medium text-stone-500 mt-1">0 elements</span>
            ) : (
              <span className="text-2xl font-bold text-stone-100">{stats.total}</span>
            )}
          </div>
          <div className="bg-stone-900 border border-stone-800 rounded-xl p-4 flex flex-col">
            <span className="text-stone-400 text-sm">Rejected</span>
            {stats.total === 0 ? (
              <span className="text-lg font-medium text-stone-500 mt-1">—</span>
            ) : (
              <span className="text-2xl font-bold text-red-400">{stats.rejectedCount}</span>
            )}
          </div>
          <div className="bg-stone-900 border border-stone-800 rounded-xl p-4 flex flex-col">
            <span className="text-stone-400 text-sm">Top Predicted</span>
            {stats.total === 0 ? (
              <span className="text-lg font-medium text-stone-500 mt-1">—</span>
            ) : (
              <span className="text-2xl font-bold text-amber-400 truncate">{stats.topClass}</span>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <section className="bg-stone-900 border border-stone-800 rounded-2xl p-5 lg:p-6 min-h-[400px] flex flex-col gap-5">
          <div className="flex flex-col gap-4 border-b border-stone-800 pb-4 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-stone-500">Inspection workspace</p>
              <h2 className="text-lg font-bold text-stone-100 mt-1">Image overlay</h2>
              <p className="text-sm text-stone-400 mt-1">Review the detected regions against the original glyph before opening annotation.</p>
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

          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-xl border border-stone-800 bg-stone-950/70 px-4 py-3">
              <div className="text-stone-500">Image</div>
              <div className="mt-1 font-medium text-stone-100 truncate">{record.imageName}</div>
            </div>
            <div className="rounded-xl border border-stone-800 bg-stone-950/70 px-4 py-3">
              <div className="text-stone-500">Overlay focus</div>
              <div className="mt-1 font-medium text-stone-100">
                {focusedIdx !== null 
                  ? `${focusedIdx} (${(record.annotations ?? {})[focusedIdx] || record.result.elements[focusedIdx]?.class_name || 'Unknown'})` 
                  : 'None selected'}
              </div>
            </div>
          </div>

          <div className="bg-stone-950 border border-stone-800 rounded-2xl overflow-hidden relative flex-1 flex items-center justify-center min-h-[340px]">
            <div className="relative inline-block max-w-full p-3">
              <img
                ref={imageRef}
                src={record.imageDataUrl}
                alt={record.imageName}
                className="block max-w-full max-h-[70vh] object-contain rounded-lg"
              />
              <canvas
                ref={canvasRef}
                className="absolute top-3 left-3 pointer-events-none"
              />
            </div>
          </div>
        </section>

        <section className="bg-stone-900 border border-stone-800 rounded-2xl p-5 lg:p-6 flex flex-col h-[70vh] overflow-hidden">
          <div className="flex items-start justify-between gap-4 border-b border-stone-800 pb-4 mb-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-stone-500">Proposal panel</p>
              <h2 className="text-lg font-bold mt-1">Detected Elements</h2>
              <p className="text-sm text-stone-400 mt-1">Browse every proposal, expand alternatives, and lock focus for the overlay.</p>
            </div>
            <div className="rounded-xl border border-stone-800 bg-stone-950/70 px-3 py-2 text-right">
              <div className="text-xs text-stone-500">Visible proposals</div>
              <div className="text-lg font-semibold text-stone-100">{record.result.elements.length}</div>
            </div>
          </div>

          {stats && stats.rejectedCount === stats.total && stats.total > 0 && (
            <div className="mb-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-xl flex items-start gap-3">
              <Info className="text-amber-500 shrink-0 mt-0.5" size={18} />
              <div>
                <p className="text-sm text-stone-200">All proposals were rejected — consider annotating to improve results</p>
                <Link to={`/annotate/${record.id}`} className="text-xs text-amber-500 hover:text-amber-400 font-medium mt-1 inline-block">Go to Annotation →</Link>
              </div>
            </div>
          )}

          <div className="overflow-y-auto pr-2 space-y-3 flex-1 custom-scrollbar">
            {record.result.elements.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-6 bg-stone-950/50 rounded-xl border border-stone-800 border-dashed">
                <Info className="text-stone-600 mb-3" size={32} />
                <p className="text-stone-400">No elements detected — try a different image or check segmentation settings</p>
              </div>
            ) : (
              record.result.elements.map((el, idx) => {
                const hasAnnotation = (record.annotations ?? {})[idx] !== undefined;
              const displayClass = hasAnnotation ? (record.annotations ?? {})[idx] : el.class_name;
              const isExpanded = expandedItems[idx];
              const isFocused = focusedIdx === idx;
              const isHovered = hoveredIdx === idx;
              const badgeClasses = el.rejected 
                ? "bg-red-400/10 text-red-400 border border-red-400/20" 
                : "bg-amber-400/10 text-amber-400 border border-amber-400/20";
              const idxBadgeClasses = el.rejected ? "bg-red-500 text-stone-950" : "bg-amber-500 text-stone-950";

              const [, , bw, bh] = el.bbox;
              let destW = 80;
              let destH = 80;
              if (bw >= bh) {
                destH = Math.max(1, 80 * (bh / bw));
              } else {
                destW = Math.max(1, 80 * (bw / bh));
              }

              return (
                <div
                  key={idx}
                  className={`bg-stone-950 border rounded-xl overflow-hidden transition-colors ${isFocused ? 'border-amber-500/60 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]' : isHovered ? 'border-stone-700' : 'border-stone-800'}`}
                  onMouseEnter={() => setHoveredIdx(idx)}
                  onMouseLeave={() => setHoveredIdx((current) => current === idx ? null : current)}
                >
                  <div 
                    className={`p-3 flex items-center justify-between cursor-pointer transition-colors ${isFocused ? 'bg-amber-500/5' : 'hover:bg-stone-800/50'}`}
                    onClick={() => {
                      setFocusedIdx((current) => current === idx ? null : idx);
                      toggleExpand(idx);
                    }}
                    onFocus={() => setFocusedIdx(idx)}
                    onBlur={() => setFocusedIdx((current) => current === idx ? null : current)}
                    tabIndex={0}
                    role="button"
                  >
                    <div className="flex items-center gap-3">
                      <div className="shrink-0 flex items-center justify-center w-[80px] h-[80px] bg-stone-900 rounded-lg overflow-hidden border border-stone-800/50">
                        <canvas
                          ref={(c) => { cropCanvasRefs.current[idx] = c; }}
                          width={destW}
                          height={destH}
                          className="block rounded bg-stone-800"
                        />
                      </div>
                      <span className={`w-8 h-8 flex items-center justify-center rounded-md font-bold text-sm shrink-0 ${idxBadgeClasses}`}>
                        {idx}
                      </span>
                      <div className="flex flex-col ml-1">
                        <span className="font-semibold text-stone-100">{displayClass}</span>
                        {hasAnnotation && (
                          <span className="text-xs text-green-400 flex items-center gap-1 mt-0.5">
                            <CheckCircle2 size={12} /> Corrected
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-3 ml-2">
                      {isFocused && <span className="text-xs font-medium uppercase tracking-wide text-amber-400 hidden sm:inline-block">Focused</span>}
                      <span className={`text-sm px-2 py-1 rounded-md font-medium shrink-0 ${badgeClasses}`}>
                        {(el.confidence * 100).toFixed(1)}%
                      </span>
                      {isExpanded ? <ChevronUp size={18} className="text-stone-500 shrink-0" /> : <ChevronDown size={18} className="text-stone-500 shrink-0" />}
                    </div>
                  </div>
                  
                  {isExpanded && (
                    <div className="p-3 pt-0 border-t border-stone-800/50 bg-stone-950/50">
                      <div className="mt-3 text-sm text-stone-400 mb-2 font-medium">Alternative predictions:</div>
                      <div className="space-y-2">
                        {el.top_k.map((tk, kIdx) => (
                          <div key={kIdx} className="flex items-center justify-between text-sm">
                            <span className="text-stone-300">{tk.class_name}</span>
                            <span className="text-stone-500">{(tk.confidence * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                      {el.rejected && (
                        <div className="mt-3 flex items-start gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg p-2">
                          <AlertCircle size={14} className="mt-0.5 shrink-0" />
                          <span className="text-xs">Low confidence prediction. Review recommended.</span>
                        </div>
                      )}
                      
                      {(isFocused || el.rejected) && (
                        <div className="mt-4 flex justify-end">
                          <Link
                            to={`/annotate/${record.id}`}
                            className={`flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors ${
                              el.rejected
                                ? 'bg-red-500 hover:bg-red-400 text-stone-950'
                                : 'bg-amber-500 hover:bg-amber-400 text-stone-950'
                            }`}
                          >
                            <Edit3 size={14} />
                            {el.rejected ? 'Correct this element' : 'Annotate'}
                          </Link>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            }))}
          </div>
        </section>
      </div>
    </div>
  );
}
