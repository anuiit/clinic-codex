import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Loader2, AlertCircle } from 'lucide-react';
import { segmentGlyph } from '../services/api';
import { saveAnalysis } from '../services/storage';
import type { AnalysisRecord } from '../types';

export default function UploadPage() {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const currentFileRef = useRef<File | null>(null);
  const navigate = useNavigate();

  const handleFile = (f: File) => {
    currentFileRef.current = f;
    setFile(f);
    setPreview(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      if (currentFileRef.current === f) {
        setPreview(e.target?.result as string);
      }
    };
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const analyze = async () => {
    if (!file || !preview) return;
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
      navigate(`/analysis/${record.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'API error — is the Flask server running on port 5000?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-2">Analyze a Codex Glyph</h1>
      <p className="text-stone-400 mb-6">Upload a glyph image to segment and identify its Nahuatl elements.</p>

      <div
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${dragging ? 'border-amber-400 bg-amber-400/5' : 'border-stone-700 hover:border-stone-500'}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/bmp,image/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
        {preview ? (
          <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg object-contain" />
        ) : (
          <div className="flex flex-col items-center gap-3 text-stone-400">
            <Upload size={40} />
            <p>Drop a glyph image here or click to browse</p>
            <p className="text-sm">PNG, JPG, BMP supported</p>
          </div>
        )}
      </div>

      {file && (
        <div className="mt-3 text-sm text-stone-400">{file.name} — {(file.size / 1024).toFixed(0)} KB</div>
      )}

      {error && (
        <div className="mt-4 flex items-start gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg p-3">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      <button
        onClick={analyze}
        disabled={!file || !preview || loading}
        className="mt-6 w-full py-3 px-6 bg-amber-500 hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed text-stone-950 font-semibold rounded-lg flex items-center justify-center gap-2 transition-colors"
      >
        {loading ? <><Loader2 size={18} className="animate-spin" /> Analyzing…</> : 'Analyze Glyph'}
      </button>
    </div>
  );
}
