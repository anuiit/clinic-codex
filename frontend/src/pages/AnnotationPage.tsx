import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { useParams, useNavigate, Link, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Save, Loader2, ZoomIn, ZoomOut, Maximize2, PenTool, MousePointer2, Trash2 } from 'lucide-react';
import { getAnalysisById, updateElements } from '../services/storage';
import { getClasses } from '../services/api';
import { appText } from '../i18n/text';
import type { AnalysisRecord, DetectedElement } from '../types';

type DragState = {
  type: 'draw' | 'move' | 'resize';
  idx: number;
  corner?: 'tl' | 'tr' | 'bl' | 'br';
  startX: number;
  startY: number;
  origBbox?: [number, number, number, number];
} | null;

export default function AnnotationPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const cardRefs = useRef<Array<HTMLDivElement | null>>([]);

  const elementParam = searchParams.get('element');
  const initialFocusedIdx = elementParam !== null && !Number.isNaN(Number(elementParam))
    ? Number(elementParam)
    : null;
  
  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [elements, setElements] = useState<DetectedElement[]>([]);
  const [classes, setClasses] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const [focusedIdx, setFocusedIdx] = useState<number | null>(initialFocusedIdx);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [dragState, setDragState] = useState<DragState>(null);
  const [tempBbox, setTempBbox] = useState<[number, number, number, number] | null>(null);
  
  const imageRef = useRef<HTMLImageElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const pendingTempBboxRef = useRef<[number, number, number, number] | null>(null);
  const t = appText.annotation;

  const getSvgCoordinates = (e: ReactPointerEvent<SVGSVGElement>) => {
    const pt = e.currentTarget.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = e.currentTarget.getScreenCTM();
    if (!ctm) return null;
    const svgPt = pt.matrixTransform(ctm.inverse());
    return { x: svgPt.x, y: svgPt.y };
  };

  const scheduleTempBbox = (bbox: [number, number, number, number]) => {
    pendingTempBboxRef.current = bbox;
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
    }
    rafRef.current = requestAnimationFrame(() => {
      if (pendingTempBboxRef.current) {
        setTempBbox(pendingTempBboxRef.current);
      }
      rafRef.current = null;
    });
  };

  useEffect(() => {
    async function loadData() {
      if (!id) { setLoading(false); return; }
      const rec = getAnalysisById(id);
      setRecord(rec);

      if (!rec) { setLoading(false); return; }
      // Deep copy elements so we can mutate safely
      setElements(JSON.parse(JSON.stringify(rec.result.elements)));

      try {
        const classesResult = await getClasses();
        setClasses(classesResult.class_names);
      } catch {
        // failed to load classes
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [id]);

  useEffect(() => {
    if (!record || !imageRef.current || !previewCanvasRef.current || focusedIdx === null) return;
    const el = elements[focusedIdx];
    if (!el) return;
    
    const [x, y, w, h] = dragState?.idx === focusedIdx && tempBbox ? tempBbox : el.bbox;
    const canvas = previewCanvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imageRef.current.complete) {
      ctx.imageSmoothingEnabled = false;
      const scaledW = w * 3;
      const scaledH = h * 3;
      const drawX = (canvas.width - scaledW) / 2;
      const drawY = (canvas.height - scaledH) / 2;
      ctx.drawImage(imageRef.current, x, y, w, h, drawX, drawY, scaledW, scaledH);
    }
  }, [record, elements, focusedIdx, tempBbox, dragState]);

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  const getHitHandle = (x: number, y: number, bbox: [number, number, number, number]) => {
    const [bx, by, bw, bh] = bbox;
    const hSize = 12; // slightly larger hit area
    if (Math.abs(x - bx) <= hSize && Math.abs(y - by) <= hSize) return 'tl';
    if (Math.abs(x - (bx + bw)) <= hSize && Math.abs(y - by) <= hSize) return 'tr';
    if (Math.abs(x - bx) <= hSize && Math.abs(y - (by + bh)) <= hSize) return 'bl';
    if (Math.abs(x - (bx + bw)) <= hSize && Math.abs(y - (by + bh)) <= hSize) return 'br';
    return null;
  };

  const getHandleHit = (x: number, y: number) => {
    let bestHit: { idx: number; corner: 'tl' | 'tr' | 'bl' | 'br'; area: number } | null = null;
    for (const [idx, el] of elements.entries()) {
      const handle = getHitHandle(x, y, el.bbox);
      if (!handle) continue;

      const [, , bw, bh] = el.bbox;
      const area = bw * bh;
      if (!bestHit || area < bestHit.area) {
        bestHit = { idx, corner: handle, area };
      }
    }

    return bestHit;
  };

  const handleSvgPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;
    const coords = getSvgCoordinates(e);
    if (!coords) return;
    const { x, y } = coords;

    if (drawMode) {
      const bbox: [number, number, number, number] = [x, y, 0, 0];
      setDragState({ type: 'draw', idx: elements.length, startX: x, startY: y });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const handleHit = getHandleHit(x, y);

    if (handleHit) {
      const el = elements[handleHit.idx];
      const bbox: [number, number, number, number] = [...el.bbox];
      setFocusedIdx(handleHit.idx);
      setDragState({ type: 'resize', idx: handleHit.idx, corner: handleHit.corner, startX: x, startY: y, origBbox: bbox });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    let hitIdx: number | null = null;
    let minArea = Infinity;
    elements.forEach((el, idx) => {
      const [bx, by, bw, bh] = el.bbox;
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
        const area = bw * bh;
        if (area < minArea) {
          minArea = area;
          hitIdx = idx;
        }
      }
    });

    if (hitIdx !== null) {
      const el = elements[hitIdx];
      const bbox: [number, number, number, number] = [...el.bbox];
      setFocusedIdx(hitIdx);
      setDragState({ type: 'move', idx: hitIdx, startX: x, startY: y, origBbox: bbox });
      setTempBbox(bbox);
      pendingTempBboxRef.current = bbox;
      e.currentTarget.setPointerCapture(e.pointerId);
    } else {
      setFocusedIdx(null);
    }
  };

  const handleSvgPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!record) return;
    const coords = getSvgCoordinates(e);
    if (!coords) return;
    const { x, y } = coords;
    const [origW, origH] = record.result.image_size;

    if (dragState) {
      let nextBbox: [number, number, number, number] | null = null;
      if (dragState.type === 'draw') {
        const minX = Math.min(dragState.startX, x);
        const minY = Math.min(dragState.startY, y);
        const maxX = Math.max(dragState.startX, x);
        const maxY = Math.max(dragState.startY, y);
        nextBbox = [
          Math.max(0, minX),
          Math.max(0, minY),
          Math.min(origW - Math.max(0, minX), maxX - minX),
          Math.min(origH - Math.max(0, minY), maxY - minY)
        ];
      } else if (dragState.type === 'move' && dragState.origBbox) {
        const dx = x - dragState.startX;
        const dy = y - dragState.startY;
        const [origBx, origBy, bw, bh] = dragState.origBbox;
        const bx = Math.max(0, Math.min(origW - bw, origBx + dx));
        const by = Math.max(0, Math.min(origH - bh, origBy + dy));
        nextBbox = [bx, by, bw, bh];
      } else if (dragState.type === 'resize' && dragState.origBbox && dragState.corner) {
        let [bx, by, bw, bh] = dragState.origBbox;
        if (dragState.corner === 'tl') {
          const nx = Math.min(bx + bw - 1, Math.max(0, x));
          const ny = Math.min(by + bh - 1, Math.max(0, y));
          bw = bx + bw - nx;
          bh = by + bh - ny;
          bx = nx;
          by = ny;
        } else if (dragState.corner === 'tr') {
          const ny = Math.min(by + bh - 1, Math.max(0, y));
          bw = Math.min(origW - bx, Math.max(1, x - bx));
          bh = by + bh - ny;
          by = ny;
        } else if (dragState.corner === 'bl') {
          const nx = Math.min(bx + bw - 1, Math.max(0, x));
          bw = bx + bw - nx;
          bx = nx;
          bh = Math.min(origH - by, Math.max(1, y - by));
        } else if (dragState.corner === 'br') {
          bw = Math.min(origW - bx, Math.max(1, x - bx));
          bh = Math.min(origH - by, Math.max(1, y - by));
        }
        nextBbox = [bx, by, bw, bh];
      }

      if (nextBbox) {
        scheduleTempBbox(nextBbox);
      }
    } else if (!drawMode) {
      let hitIdx: number | null = null;
      let minArea = Infinity;
      elements.forEach((el, idx) => {
        const [bx, by, bw, bh] = el.bbox;
        if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
          const area = bw * bh;
          if (area < minArea) {
            minArea = area;
            hitIdx = idx;
          }
        }
      });
      setHoveredIdx(hitIdx);
      
      const svg = e.currentTarget;
      let cursor = 'default';
      if (focusedIdx !== null) {
        const el = elements[focusedIdx];
        if (el) {
          const handle = getHitHandle(x, y, el.bbox);
          if (handle === 'tl' || handle === 'br') cursor = 'nwse-resize';
          else if (handle === 'tr' || handle === 'bl') cursor = 'nesw-resize';
        }
      }
      if (cursor === 'default' && hitIdx !== null) {
        cursor = 'move';
      }
      svg.style.cursor = cursor;
    } else {
      e.currentTarget.style.cursor = 'crosshair';
    }
  };

  const handleSvgPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    const coords = getSvgCoordinates(e);
    if (!coords) return;

    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const finalTempBbox = pendingTempBboxRef.current ?? tempBbox;
    if (!dragState || !finalTempBbox) {
      pendingTempBboxRef.current = null;
      return;
    }

    const newElements = [...elements];
    if (dragState.type === 'draw') {
      if (finalTempBbox[2] > 5 && finalTempBbox[3] > 5) {
        newElements.push({
          bbox: finalTempBbox,
          class_name: 'unknown',
          class_label: 0,
          confidence: 1.0,
          top_k: [],
          rejected: false
        });
        setFocusedIdx(newElements.length - 1);
      }
    } else if ((dragState.type === 'move' || dragState.type === 'resize') && dragState.idx < newElements.length) {
      newElements[dragState.idx].bbox = finalTempBbox;
    }

    setElements(newElements);
    setDragState(null);
    setTempBbox(null);
    pendingTempBboxRef.current = null;
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.key === 'Delete' || e.key === 'Backspace') && focusedIdx !== null) {
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
        setElements(prev => prev.filter((_, idx) => idx !== focusedIdx));
        setFocusedIdx(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [focusedIdx]);

  const handleSave = () => {
    if (!id) return;
    setSaving(true);
    const ok = updateElements(id, elements);
    if (!ok) {
      setSaving(false);
      return;
    }
    setSaving(false);
    setTimeout(() => {
      navigate('/');
    }, 300);
  };

  if (!record && !loading) {
    return (
      <div className="max-w-2xl mx-auto text-center py-12">
        <h2 className="text-2xl font-bold text-stone-100 mb-4">{t.notFound}</h2>
        <Link to="/" className="text-amber-400 hover:text-amber-300 inline-flex items-center gap-2">
          <ArrowLeft size={18} /> {t.backToHistory}
        </Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader2 className="animate-spin text-amber-500" size={32} />
      </div>
    );
  }

  return (
    <div className="h-full w-full overflow-hidden flex flex-col gap-2 p-2">
      <div className="flex shrink-0 items-center justify-between rounded-xl border border-stone-800 bg-stone-900/80 px-4 py-2 sidebar-header">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-stone-400 hover:text-stone-100 flex items-center gap-2 transition-colors">
            <ArrowLeft size={18} /> {t.back}
          </Link>
          <h1 className="text-lg font-bold text-stone-100">{t.title}</h1>
          
          <div className="h-6 w-px bg-stone-700 mx-2" />
          
          <button
            onClick={() => setDrawMode(!drawMode)}
            className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${drawMode ? 'bg-amber-500 text-stone-950' : 'bg-stone-800 text-stone-300 hover:bg-stone-700'}`}
          >
            {drawMode ? <PenTool size={16} /> : <MousePointer2 size={16} />}
            {drawMode ? t.drawMode : t.selectMode}
          </button>
          
          <div className="flex items-center gap-1 rounded-lg border border-stone-700 bg-stone-900 p-1">
            <button onClick={() => setZoom(z => Math.min(4, z + 0.25))} className="p-1.5 text-stone-400 hover:text-stone-100 hover:bg-stone-800 rounded">
              <ZoomIn size={16} />
            </button>
            <button onClick={() => setZoom(1)} className="p-1.5 text-stone-400 hover:text-stone-100 hover:bg-stone-800 rounded">
              <Maximize2 size={16} />
            </button>
            <button onClick={() => setZoom(z => Math.max(0.25, z - 0.25))} className="p-1.5 text-stone-400 hover:text-stone-100 hover:bg-stone-800 rounded">
              <ZoomOut size={16} />
            </button>
          </div>
        </div>
        
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 rounded-lg bg-amber-500 px-4 py-1.5 text-sm font-semibold text-stone-950 transition-colors hover:bg-amber-400 disabled:opacity-50"
        >
          {saving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} />}
          {t.saveChanges}
        </button>
      </div>

      <div className="flex min-h-0 flex-1 gap-2">
        <div className="flex-1 overflow-auto rounded-xl border border-stone-800 bg-stone-900/80 relative flex items-center justify-center" onWheel={(e) => {
          if (e.ctrlKey) {
            e.preventDefault();
            setZoom(z => Math.max(0.25, Math.min(4, z - e.deltaY * 0.01)));
          }
        }}>
          <div style={{ transform: `scale(${zoom})`, transformOrigin: 'center center', transition: 'transform 0.1s ease', willChange: 'transform' }} className="relative inline-block max-w-full max-h-full">
            <img
              ref={imageRef}
              src={record!.imageDataUrl}
              alt={record!.imageName}
              className="block max-w-full max-h-full w-auto h-auto pointer-events-none"
            />
            {record && (
              <svg
                className="absolute left-0 top-0 w-full h-full"
                viewBox={`0 0 ${record.result.image_size[0]} ${record.result.image_size[1]}`}
                style={{ width: '100%', height: '100%' }}
                onPointerDown={handleSvgPointerDown}
                onPointerMove={handleSvgPointerMove}
                onPointerUp={handleSvgPointerUp}
              >
                {elements.map((el, idx) => {
                  const [x, y, w, h] = idx === dragState?.idx && dragState.type !== 'draw' && tempBbox ? tempBbox : el.bbox;
                  const isFocused = idx === focusedIdx;
                  const isHovered = idx === hoveredIdx;
                  const strokeColor = el.rejected ? '#ef4444' : isFocused ? '#ffffff' : isHovered ? '#fbbf24' : '#f59e0b';
                  return (
                    <g key={idx}>
                      <rect x={x} y={y} width={w} height={h} fill="none" stroke={strokeColor} strokeWidth={isFocused ? 3 : 2} />
                      {isFocused && !drawMode && (
                        <>
                          <rect x={x-5} y={y-5} width={10} height={10} fill="#ffffff" className="cursor-nwse-resize" />
                          <rect x={x+w-5} y={y-5} width={10} height={10} fill="#ffffff" className="cursor-nesw-resize" />
                          <rect x={x-5} y={y+h-5} width={10} height={10} fill="#ffffff" className="cursor-nesw-resize" />
                          <rect x={x+w-5} y={y+h-5} width={10} height={10} fill="#ffffff" className="cursor-nwse-resize" />
                        </>
                      )}
                      <rect x={x} y={Math.max(0, y - 24)} width={28} height={24} fill={strokeColor} />
                      <text x={x + 14} y={Math.max(0, y - 24) + 12} fill="#0c0a09" fontSize="14" fontWeight="bold" fontFamily="sans-serif" textAnchor="middle" dominantBaseline="central">{idx}</text>
                    </g>
                  );
                })}
                {drawMode && dragState?.type === 'draw' && tempBbox && (
                  <rect x={tempBbox[0]} y={tempBbox[1]} width={tempBbox[2]} height={tempBbox[3]} fill="none" stroke="#3b82f6" strokeWidth={2} strokeDasharray="4 4" />
                )}
              </svg>
            )}
          </div>
        </div>

        <div className="w-[30%] shrink-0 flex flex-col rounded-xl border border-stone-800 bg-stone-900/80 p-4 shadow-sm sidebar-shell">
          <div className="mb-4 border border-stone-800 rounded bg-stone-950 flex justify-center items-center h-[200px] shrink-0">
            {focusedIdx !== null ? (
              <canvas ref={previewCanvasRef} width={200} height={200} className="block w-[200px] h-[200px] object-contain" />
            ) : (
              <div className="text-sm text-stone-500 text-center px-4">{t.selectElementCrop}</div>
            )}
          </div>
          
          <div className="mb-2 text-xs uppercase tracking-[0.24em] text-stone-500 sidebar-header">
            {t.elements} ({elements.length})
          </div>

          <div className="flex-1 space-y-3 overflow-y-auto pr-2 sidebar-body">
            {elements.map((el, idx) => {
              const isFocused = idx === focusedIdx;
              const classOptions = Array.from(new Set([el.class_name, ...classes]));

              return (
                <div
                  key={idx}
                  ref={(node) => { cardRefs.current[idx] = node; }}
                  onClick={() => setFocusedIdx(idx)}
                  className={`flex flex-col gap-3 rounded-xl border bg-stone-950 p-3 transition-all cursor-pointer ${isFocused ? 'border-amber-500/60 bg-amber-500/5 shadow-[0_0_0_1px_rgba(245,158,11,0.25)]' : 'border-stone-800 hover:border-stone-700'}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-3">
                      <span className="w-8 h-8 flex items-center justify-center rounded-lg bg-amber-500 text-stone-950 font-bold text-sm shrink-0">
                        {idx}
                      </span>
                      <div className="font-semibold text-stone-100 text-sm truncate">
                        {el.class_name}
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setElements(prev => prev.filter((_, i) => i !== idx));
                        if (focusedIdx === idx) setFocusedIdx(null);
                      }}
                      className="p-1.5 text-stone-500 hover:text-red-400 hover:bg-stone-800 rounded transition-colors"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  
                  {isFocused && (
                    <div className="relative mt-1">
                      <select
                        value={el.class_name}
                        onChange={(e) => {
                          const newElements = [...elements];
                          newElements[idx].class_name = e.target.value;
                          setElements(newElements);
                        }}
                        onClick={e => e.stopPropagation()}
                        className="w-full bg-stone-900 border border-stone-700 text-stone-100 rounded-lg pl-3 pr-10 py-2 text-sm outline-none focus:border-amber-500 transition-colors appearance-none"
                      >
                        {classOptions.map(c => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-stone-500">
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                        </svg>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
