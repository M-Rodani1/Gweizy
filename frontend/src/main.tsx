import React from 'react';
import ReactDOM from 'react-dom/client';
import * as Sentry from "@sentry/react";

// Ensure React is properly loaded
if (typeof React === 'undefined' || !React.Children) {
  throw new Error('React is not properly initialized. Please check your build configuration.');
}

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
    <React.StrictMode>
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
    </React.StrictMode>
  );
}
// Cache bust Sat Dec 28 14:45:00 CET 2025 - force redeploy
