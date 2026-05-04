import { useEffect, useRef, useState } from 'react';
import { useParams, useNavigate, Link, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Save, Loader2, AlertCircle } from 'lucide-react';
import { getAnalysisById, updateAnnotations } from '../services/storage';
import { getClasses } from '../services/api';
import type { AnalysisRecord } from '../types';

export default function AnnotationPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const cardRefs = useRef<Array<HTMLDivElement | null>>([]);

  const elementParam = searchParams.get('element');
  const focusedElementIdx = elementParam !== null && !Number.isNaN(Number(elementParam))
    ? Number(elementParam)
    : null;
  
  const [record, setRecord] = useState<AnalysisRecord | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [classesError, setClassesError] = useState(false);
  const [saveError, setSaveError] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  
  const [annotations, setAnnotations] = useState<Record<number, string>>({});

  useEffect(() => {
    setClassesError(false);
    setSaveError(false);
    async function loadData() {
      if (!id) { setLoading(false); return; }
      const rec = getAnalysisById(id);
      setRecord(rec);

      if (!rec) { setLoading(false); return; }

      const initialAnns: Record<number, string> = { ...(rec.annotations ?? {}) };
      rec.result.elements.forEach((el, idx) => {
        if (initialAnns[idx] == null) {
          initialAnns[idx] = el.class_name;
        }
      });
      setAnnotations(initialAnns);

      try {
        const classesResult = await getClasses();
        setClasses(classesResult.class_names);
      } catch (e) {
        console.error("Failed to load classes", e);
        setClassesError(true);
      } finally {
        setLoading(false);
      }
    }
    
    loadData();
  }, [id]);

  useEffect(() => {
    if (loading || focusedElementIdx === null || !record) {
      return;
    }

    if (focusedElementIdx < 0 || focusedElementIdx >= record.result.elements.length) {
      return;
    }

    cardRefs.current[focusedElementIdx]?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });
  }, [focusedElementIdx, loading, record]);

  if (!record && !loading) {
    return (
      <div className="max-w-2xl mx-auto text-center py-12">
        <h2 className="text-2xl font-bold text-stone-100 mb-4">Analysis not found</h2>
        <Link to="/" className="text-amber-400 hover:text-amber-300 inline-flex items-center gap-2">
          <ArrowLeft size={18} /> Back to History
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

  const handleSave = () => {
    if (!id) return;
    setSaving(true);
    setSaveError(false);
    const ok = updateAnnotations(id, annotations);
    if (!ok) {
      setSaveError(true);
      setSaving(false);
      return;
    }
    setTimeout(() => {
      navigate('/');
    }, 300);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-stone-400 hover:text-stone-100 flex items-center gap-2 transition-colors">
            <ArrowLeft size={18} /> Back to Analysis
          </Link>
          <h1 className="text-2xl font-bold">Annotate Elements</h1>
        </div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-stone-950 font-semibold py-2 px-6 rounded-lg flex items-center gap-2 transition-colors"
        >
          {saving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} />}
          Save Annotations
        </button>
      </div>

      <div className="bg-stone-900 border border-stone-800 rounded-xl p-6">
        {saveError && (
          <div className="flex items-start gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg p-3 mb-4">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span className="text-sm">Could not save — the analysis record may have been deleted. Go back to history and re-open.</span>
          </div>
        )}
        {classesError && (
          <div className="flex items-start gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg p-3 mb-4">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <span className="text-sm">Could not load the class list from the API. You can still pick from the suggestions below, but the full dropdown won't be available.</span>
          </div>
        )}
        <p className="text-stone-400 mb-6">
          Please review the following predictions and correct them if necessary.
        </p>

        <div className="space-y-6">
          {record!.result.elements.map((el, idx) => {
            const isRejected = el.rejected;
            const isFocused = idx === focusedElementIdx;
            return (
              <div
                key={idx}
                ref={(node) => {
                  cardRefs.current[idx] = node;
                }}
                className={`bg-stone-950 border rounded-lg p-5 flex flex-col md:flex-row md:items-start gap-6 transition-colors ${isFocused ? 'border-amber-500/60 ring-1 ring-amber-500/30' : 'border-stone-800'}`}
              >
                <div className="flex items-center gap-4 shrink-0">
                  <span className={`w-12 h-12 flex items-center justify-center rounded-lg text-stone-950 font-bold text-xl ${isRejected ? 'bg-red-500' : 'bg-amber-400'}`}>
                    {idx}
                  </span>
                  <div>
                    <div className="text-stone-400 text-sm">Predicted</div>
                    <div className={`font-semibold ${isRejected ? 'text-red-400' : 'text-amber-300'}`}>{el.class_name}</div>
                    <div className={`mt-1 inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${isRejected ? 'bg-red-500/10 text-red-300 border border-red-500/20' : 'bg-amber-500/10 text-amber-200 border border-amber-500/20'}`}>
                      {isRejected ? 'Rejected' : 'Detected'}
                    </div>
                  </div>
                </div>

                <div className="flex-1 space-y-4">
                  {!classesError && (
                  <div>
                    <label className="block text-sm text-stone-400 mb-2">Correct Annotation</label>
                    <select
                      value={annotations[idx] || el.class_name}
                      onChange={(e) => setAnnotations(prev => ({ ...prev, [idx]: e.target.value }))}
                      className="w-full bg-stone-900 border border-stone-700 text-stone-100 rounded-lg p-3 outline-none focus:border-amber-500 transition-colors"
                    >
                      {classes.map(c => (
                        <option key={c} value={c}>{c}</option>
                      ))}
                    </select>
                  </div>
                  )}

                  <div className="flex flex-wrap gap-2">
                    <span className="text-xs text-stone-500 py-1">Suggestions:</span>
                    {el.top_k.map((tk, kIdx) => (
                      <button
                        key={kIdx}
                        onClick={() => setAnnotations(prev => ({ ...prev, [idx]: tk.class_name }))}
                        className="text-xs px-2 py-1 bg-stone-800 hover:bg-stone-700 rounded text-stone-300 transition-colors"
                      >
                        {tk.class_name} ({(tk.confidence * 100).toFixed(0)}%)
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
