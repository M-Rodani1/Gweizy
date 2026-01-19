import React from 'react';
import ReactDOM from 'react-dom/client';
import * as Sentry from "@sentry/react";

// CRITICAL: Ensure React is globally available BEFORE any lucide-react imports
// This prevents "Cannot set properties of undefined (setting 'Children')" errors
// Must be synchronous and happen immediately
if (typeof window !== 'undefined') {
  // Set React on window IMMEDIATELY - before any other code runs
  const win = window as any;
  win.React = React;
  win.ReactDOM = ReactDOM;
  
  // Ensure React.Children exists immediately
  if (!React.Children) {
    Object.defineProperty(React, 'Children', {
      value: {
        map: React.Children.map,
        forEach: React.Children.forEach,
        count: React.Children.count,
        toArray: React.Children.toArray,
        only: React.Children.only
      },
      writable: false,
      configurable: false
    });
  }
  
  // Ensure React.createElement is available
  if (!win.React.createElement) {
    win.React.createElement = React.createElement;
  }
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
