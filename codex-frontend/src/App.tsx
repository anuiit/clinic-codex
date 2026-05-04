import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';
import DetailPage from './pages/DetailPage';
import AnnotationPage from './pages/AnnotationPage';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-stone-950 text-stone-100">
        <nav className="border-b border-stone-800 px-6 py-4 flex items-center gap-8">
          <span className="font-bold text-amber-400 text-lg tracking-wide">Codex Analyzer</span>
          <NavLink to="/" end className={({ isActive }) => isActive ? 'text-amber-400 font-medium' : 'text-stone-400 hover:text-stone-100'}>
            Upload
          </NavLink>
          <NavLink to="/dashboard" className={({ isActive }) => isActive ? 'text-amber-400 font-medium' : 'text-stone-400 hover:text-stone-100'}>
            History
          </NavLink>
        </nav>
        <main className="max-w-6xl mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/analysis/:id" element={<DetailPage />} />
            <Route path="/annotate/:id" element={<AnnotationPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
