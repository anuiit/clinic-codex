import { useEffect, useRef, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Edit3, ChevronDown, ChevronUp, AlertCircle, CheckCircle2 } from 'lucide-react';
import { getAnalysisById } from '../services/storage';
import type { AnalysisRecord } from '../types';

export default function DetailPage() {
  const { id } = useParams<{ id: string }>();
  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [expandedItems, setExpandedItems] = useState<Record<number, boolean>>({});
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (id) {
      setRecord(getAnalysisById(id));
    }
  }, [id]);

  useEffect(() => {
    if (!record || !canvasRef.current || !imageRef.current) return;

    const drawBoxes = () => {
      const canvas = canvasRef.current;
      const image = imageRef.current;
      if (!canvas || !image) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const { width, height } = image.getBoundingClientRect();
      canvas.width = width;
      canvas.height = height;

      ctx.clearRect(0, 0, width, height);

      const [origW, origH] = record.result.image_size;
      const scaleX = width / origW;
      const scaleY = height / origH;

      record.result.elements.forEach((el, idx) => {
        const [x, y, w, h] = el.bbox;
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledW = w * scaleX;
        const scaledH = h * scaleY;

        ctx.strokeStyle = el.rejected ? '#ef4444' : '#f59e0b';
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

        ctx.fillStyle = el.rejected ? '#ef4444' : '#f59e0b';
        ctx.fillRect(scaledX, Math.max(0, scaledY - 24), 28, 24);
        
        ctx.fillStyle = '#0c0a09';
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${idx}`, scaledX + 14, Math.max(0, scaledY - 24) + 12);
      });
    };

    if (imageRef.current.complete) {
      drawBoxes();
    }

    imageRef.current.addEventListener('load', drawBoxes);
    window.addEventListener('resize', drawBoxes);

    return () => {
      imageRef.current?.removeEventListener('load', drawBoxes);
      window.removeEventListener('resize', drawBoxes);
    };
  }, [record]);

  const stats = useMemo(() => {
    if (!record) return null;
    const elements = record.result.elements;
    const rejectedCount = elements.filter(e => e.rejected).length;
    
    const classCounts: Record<string, number> = {};
    elements.forEach(e => {
      const finalClass = record.annotations[elements.indexOf(e)] || e.class_name;
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
            <span className="text-2xl font-bold text-stone-100">{stats.total}</span>
          </div>
          <div className="bg-stone-900 border border-stone-800 rounded-xl p-4 flex flex-col">
            <span className="text-stone-400 text-sm">Rejected</span>
            <span className="text-2xl font-bold text-red-400">{stats.rejectedCount}</span>
          </div>
          <div className="bg-stone-900 border border-stone-800 rounded-xl p-4 flex flex-col">
            <span className="text-stone-400 text-sm">Top Predicted</span>
            <span className="text-2xl font-bold text-amber-400 truncate">{stats.topClass}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-stone-900 border border-stone-800 rounded-xl p-4 overflow-hidden relative flex items-center justify-center min-h-[400px]">
          <div className="relative inline-block max-w-full">
            <img 
              ref={imageRef}
              src={record.imageDataUrl} 
              alt={record.imageName}
              className="max-w-full max-h-[70vh] object-contain rounded-lg"
            />
            <canvas 
              ref={canvasRef}
              className="absolute top-0 left-0 pointer-events-none"
            />
          </div>
        </div>

        <div className="bg-stone-900 border border-stone-800 rounded-xl p-4 flex flex-col h-[70vh]">
          <h2 className="text-lg font-bold mb-4 px-2">Detected Elements</h2>
          <div className="overflow-y-auto pr-2 space-y-3 flex-1 custom-scrollbar">
            {record.result.elements.map((el, idx) => {
              const hasAnnotation = record.annotations[idx] !== undefined;
              const displayClass = hasAnnotation ? record.annotations[idx] : el.class_name;
              const isExpanded = expandedItems[idx];
              const badgeClasses = el.rejected 
                ? "bg-red-400/10 text-red-400 border border-red-400/20" 
                : "bg-amber-400/10 text-amber-400 border border-amber-400/20";
              const idxBadgeClasses = el.rejected ? "bg-red-500 text-stone-950" : "bg-amber-500 text-stone-950";

              return (
                <div key={idx} className="bg-stone-950 border border-stone-800 rounded-lg overflow-hidden">
                  <div 
                    className="p-3 flex items-center justify-between cursor-pointer hover:bg-stone-800/50 transition-colors"
                    onClick={() => toggleExpand(idx)}
                  >
                    <div className="flex items-center gap-3">
                      <span className={`w-8 h-8 flex items-center justify-center rounded-md font-bold text-sm ${idxBadgeClasses}`}>
                        {idx}
                      </span>
                      <div className="flex flex-col">
                        <span className="font-semibold">{displayClass}</span>
                        {hasAnnotation && (
                          <span className="text-xs text-green-400 flex items-center gap-1 mt-0.5">
                            <CheckCircle2 size={12} /> Corrected
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`text-sm px-2 py-1 rounded-md font-medium ${badgeClasses}`}>
                        {(el.confidence * 100).toFixed(1)}%
                      </span>
                      {isExpanded ? <ChevronUp size={18} className="text-stone-500" /> : <ChevronDown size={18} className="text-stone-500" />}
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
