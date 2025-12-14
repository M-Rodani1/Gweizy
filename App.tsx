import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Docs from './pages/Docs';
import Terms from './pages/legal/Terms';
import Privacy from './pages/legal/Privacy';
import About from './pages/legal/About';

function App() {
  // Signal to Base that the mini app is ready (only in Farcaster context)
  useEffect(() => {
    const initFarcaster = async () => {
      try {
        // Only load SDK in Farcaster environment
        if (typeof window !== 'undefined' && window.parent !== window) {
          const { default: sdk } = await import('@farcaster/miniapp-sdk');
          sdk.actions.ready();
        }
      } catch (error) {
        console.warn('Farcaster SDK not available:', error);
      }
    };

    initFarcaster();
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<Dashboard />} />
        <Route path="/docs" element={<Docs />} />
        <Route path="/terms" element={<Terms />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
