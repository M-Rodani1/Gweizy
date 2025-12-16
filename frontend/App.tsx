import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { FarcasterProvider } from './src/contexts/FarcasterContext';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Docs from './pages/Docs';
import Terms from './pages/legal/Terms';
import Privacy from './pages/legal/Privacy';
import About from './pages/legal/About';

function App() {
  return (
    <FarcasterProvider>
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
    </FarcasterProvider>
  );
}

export default App;
