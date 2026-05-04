import { useEffect } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom';
import AnnotationPage from './pages/AnnotationPage';
import WorkspacePage from './pages/WorkspacePage';

function LegacyAnalysisRedirect() {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();

  useEffect(() => {
    if (id) {
      navigate(`/?analysis=${encodeURIComponent(id)}`, { replace: true });
      return;
    }

    navigate('/', { replace: true });
  }, [id, navigate]);

  return null;
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-stone-950 text-stone-100">
        <nav className="border-b border-stone-800/80 bg-stone-950/90 px-6 py-4 backdrop-blur">
          <span className="font-bold text-amber-400 text-lg tracking-wide">Codex Analyzer</span>
        </nav>
        <main className="mx-auto max-w-[1700px] px-4 py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<WorkspacePage />} />
            <Route path="/dashboard" element={<Navigate to="/" replace />} />
            <Route path="/analysis/:id" element={<LegacyAnalysisRedirect />} />
            <Route path="/annotate/:id" element={<AnnotationPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
