import React, { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { FarcasterProvider } from './src/contexts/FarcasterContext';
import { CardSkeleton } from './src/components/LoadingSkeleton';
import Landing from './pages/Landing';

// Lazy load heavy components for code splitting
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Docs = lazy(() => import('./pages/Docs'));
const Terms = lazy(() => import('./pages/legal/Terms'));
const Privacy = lazy(() => import('./pages/legal/Privacy'));
const About = lazy(() => import('./pages/legal/About'));

function App() {
  return (
    <FarcasterProvider>
      <BrowserRouter>
        <Suspense fallback={<CardSkeleton />}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/app" element={<Dashboard />} />
            <Route path="/docs" element={<Docs />} />
            <Route path="/terms" element={<Terms />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </FarcasterProvider>
  );
}

export default App;
