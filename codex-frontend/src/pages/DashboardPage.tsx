import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Trash2, Search } from 'lucide-react';
import { getHistory, deleteAnalysis } from '../services/storage';
import type { AnalysisRecord } from '../types';

export default function DashboardPage() {
  const [records, setRecords] = useState<AnalysisRecord[]>(() => getHistory());
  const [filter, setFilter] = useState('');

  const filtered = useMemo(() => {
    if (!filter) return records;
    const q = filter.toLowerCase();
    return records.filter((r) => {
      const names = r.result.elements.map((e) => e.class_name.toLowerCase());
      const annotated = Object.values(r.annotations ?? {}).map((v) => v.toLowerCase());
      return names.some((n) => n.includes(q)) || annotated.some((a) => a.includes(q)) || r.imageName.toLowerCase().includes(q);
    });
  }, [records, filter]);

  const remove = (id: string) => {
    deleteAnalysis(id);
    setRecords(getHistory());
  };

  if (records.length === 0) {
    return (
      <div className="text-center py-24 text-stone-500">
        <p className="text-lg">No analyses yet.</p>
        <Link to="/" className="mt-4 inline-block text-amber-400 hover:underline">Upload your first glyph →</Link>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Analysis History</h1>
        <div className="relative">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by glyph…"
            className="pl-9 pr-4 py-2 bg-stone-900 border border-stone-700 rounded-lg text-sm focus:outline-none focus:border-amber-400"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {filtered.map((r) => (
          <div key={r.id} className="relative bg-stone-900 border border-stone-800 rounded-xl overflow-hidden hover:border-stone-600 transition-colors group">
            <Link to={`/analysis/${r.id}`}>
              <img src={r.imageDataUrl} alt={r.imageName} className="w-full h-40 object-cover" />
              <div className="p-3">
                <p className="font-medium text-sm truncate">{r.imageName}</p>
                <p className="text-xs text-stone-400 mt-1">{r.result.num_elements} elements detected</p>
                <div className="flex flex-wrap gap-1 mt-2">
                  {[...new Set(r.result.elements.map((e, i) => ({ e, i })).filter(({ e }) => !e.rejected).map(({ e, i }) => r.annotations?.[i] ?? e.class_name))].slice(0, 3).map(name => (
                    <span key={name} className="text-xs bg-amber-400/10 text-amber-300 px-1.5 py-0.5 rounded">{name}</span>
                  ))}
                  {r.result.elements.some(e => e.rejected) && (
                    <span className="text-xs bg-red-400/10 text-red-300 px-1.5 py-0.5 rounded">
                      {r.result.elements.filter(e => e.rejected).length} rejected
                    </span>
                  )}
                </div>
                <p className="text-xs text-stone-600 mt-2">{new Date(r.timestamp).toLocaleDateString()}</p>
              </div>
            </Link>
            <button
              onClick={() => remove(r.id)}
              className="opacity-0 group-hover:opacity-100 absolute top-2 right-2 p-1 bg-stone-800 rounded text-red-400 hover:text-red-300 transition-opacity"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
