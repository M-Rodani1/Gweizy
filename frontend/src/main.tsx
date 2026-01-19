import React from 'react';
import ReactDOM from 'react-dom/client';
import * as Sentry from "@sentry/react";

// CRITICAL: Ensure React is globally available BEFORE any lucide-react imports
// This prevents "Cannot set properties of undefined (setting 'Children')" errors
if (typeof window !== 'undefined') {
  (window as any).React = React;
  // Ensure React.createElement and other core methods are available
  (window as any).ReactDOM = ReactDOM;
}

// Import lucide-react fix utility
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
