import React from 'react';
import ReactDOM from 'react-dom/client';
import * as Sentry from "@sentry/react";

// CRITICAL: Ensure React is globally available BEFORE any lucide-react imports
// This prevents "Cannot set properties of undefined (setting 'Children')" errors
// Must be synchronous and happen immediately - BEFORE any component imports
if (typeof window !== 'undefined') {
  // Set React on window IMMEDIATELY - before any other code runs
  const win = window as any;
  
  // Make React available globally FIRST
  win.React = React;
  win.ReactDOM = ReactDOM;
  
  // Ensure React.createElement is available immediately
  win.React.createElement = React.createElement;
  
  // Ensure React.Children exists and is properly set up
  // This is critical for lucide-react which tries to set React.Children
  if (!win.React.Children) {
    win.React.Children = React.Children;
  }
  
  // Ensure all React core exports are available
  Object.keys(React).forEach(key => {
    if (!win.React[key]) {
      win.React[key] = (React as any)[key];
    }
  });
  
  // Force React to be fully initialized
  // Some libraries check for React.Children specifically
  if (!React.Children || typeof React.Children !== 'object') {
    console.warn('React.Children is not properly initialized');
  }
  
  // Mark React as ready for lucide-react
  win.__REACT_READY__ = true;
}

// Import lucide-react fix utility AFTER React is set up
import './utils/lucideReactFix';

import App from '../App';
import ErrorBoundary from './components/ErrorBoundary';
import { QueryProvider } from './providers/QueryProvider';
import { ToastProvider } from './providers/ToastProvider';
import { ChainProvider } from './contexts/ChainContext';
import { SchedulerProvider } from './contexts/SchedulerContext';
import { PreferencesProvider } from './contexts/PreferencesContext';
import { OfflineIndicator } from './components/OfflineIndicator';
import { registerServiceWorker } from './utils/registerSW';
import './index.css';

// Initialize Sentry for error tracking
const sentryDsn = import.meta.env.VITE_SENTRY_DSN;
if (sentryDsn) {
  Sentry.init({
    dsn: sentryDsn,
    integrations: [
      Sentry.browserTracingIntegration(),
      Sentry.replayIntegration(),
    ],
    tracesSampleRate: 0.1,
    replaysSessionSampleRate: 0.1,
    replaysOnErrorSampleRate: 1.0,
    environment: import.meta.env.MODE
  });
}

// Register service worker for caching
registerServiceWorker();

const root = document.getElementById('root');
if (root) {
  ReactDOM.createRoot(root).render(
    // StrictMode disabled to avoid lucide-react compatibility issues
    <ErrorBoundary>
      <QueryProvider>
        <PreferencesProvider>
          <ChainProvider>
            <SchedulerProvider>
              <ToastProvider>
                <OfflineIndicator />
                <App />
              </ToastProvider>
            </SchedulerProvider>
          </ChainProvider>
        </PreferencesProvider>
      </QueryProvider>
    </ErrorBoundary>
  );
}
// Cache bust Sat Dec 28 14:45:00 CET 2025 - force redeploy
