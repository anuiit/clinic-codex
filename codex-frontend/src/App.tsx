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
      <div className="h-screen w-screen overflow-hidden flex flex-col bg-stone-950 text-stone-100 p-4">
        <main className="flex-1 overflow-hidden rounded-xl">
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
