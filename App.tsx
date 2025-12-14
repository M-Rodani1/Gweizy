import React, { useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Docs from './pages/Docs';
import Terms from './pages/legal/Terms';
import Privacy from './pages/legal/Privacy';
import About from './pages/legal/About';
import sdk from '@farcaster/miniapp-sdk';

function App() {
  // Signal to Base that the mini app is ready
  useEffect(() => {
    try {
      sdk.actions.ready();
    } catch (error) {
      console.warn('Farcaster SDK not available:', error);
    }
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
