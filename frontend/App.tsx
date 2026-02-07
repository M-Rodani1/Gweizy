import React, { lazy, Suspense, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { FarcasterProvider } from './src/contexts/FarcasterContext';
import { CardSkeleton } from './src/components/LoadingSkeleton';
import Landing from './pages/Landing';

// Lazy load heavy components for code splitting with prefetch hints
const Dashboard = lazy(() => import(/* webpackChunkName: "dashboard" */ './pages/Dashboard'));
const Pricing = lazy(() => import(/* webpackChunkName: "pricing" */ './pages/Pricing'));
const Analytics = lazy(() => import(/* webpackChunkName: "analytics" */ './pages/Analytics'));
const SystemStatus = lazy(() => import(/* webpackChunkName: "system" */ './pages/SystemStatus'));
const Terms = lazy(() => import('./pages/legal/Terms'));
const Privacy = lazy(() => import('./pages/legal/Privacy'));
const About = lazy(() => import('./pages/legal/About'));

// Prefetch Dashboard when user is on Landing page (most common navigation)
const usePrefetchDashboard = () => {
  const location = useLocation();
  
  useEffect(() => {
    if (location.pathname === '/') {
      // Prefetch Dashboard after 2 seconds on landing page
      const timer = setTimeout(() => {
        import('./pages/Dashboard');
      }, 2000);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [location.pathname]);
};

// Component to trigger prefetch
const PrefetchManager: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  usePrefetchDashboard();
  return <>{children}</>;
};

function App() {
  return (
    <FarcasterProvider>
      <BrowserRouter>
        <PrefetchManager>
          <Suspense fallback={<CardSkeleton />}>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/app" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/system" element={<SystemStatus />} />
              <Route path="/pricing" element={<Pricing />} />
              <Route path="/terms" element={<Terms />} />
              <Route path="/privacy" element={<Privacy />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </Suspense>
        </PrefetchManager>
      </BrowserRouter>
    </FarcasterProvider>
  );
}

export default App;
