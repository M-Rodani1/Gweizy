import React from 'react';
import ReactDOM from 'react-dom/client';
import * as Sentry from "@sentry/react";
import App from '../App';
import ErrorBoundary from './components/ErrorBoundary';
import { QueryProvider } from './providers/QueryProvider';
import { ToastProvider } from './providers/ToastProvider';
import { ChainProvider } from './contexts/ChainContext';
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
          <ChainProvider>
            <ToastProvider>
              <OfflineIndicator />
              <App />
            </ToastProvider>
          </ChainProvider>
        </QueryProvider>
      </ErrorBoundary>
    </React.StrictMode>
  );
}
// Cache bust Thu Dec 18 22:30:54 CET 2025
