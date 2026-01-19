import React from 'react';
import ReactDOM from 'react-dom/client';
import * as Sentry from "@sentry/react";

// CRITICAL: Ensure React is globally available BEFORE any lucide-react imports
// This prevents "Cannot set properties of undefined (setting 'Children')" errors
// Must be synchronous and happen immediately - BEFORE any component imports
if (typeof window !== 'undefined') {
  const win = window as any;
  
  // Make React available globally FIRST - before any imports execute
  win.React = React;
  win.ReactDOM = ReactDOM;
  
  // CRITICAL: Ensure React.Children is available IMMEDIATELY
  // lucide-react tries to access React.Children, so it must exist
  if (React.Children) {
    win.React.Children = React.Children;
  }
  
  // Ensure React.createElement is available
  win.React.createElement = React.createElement;
  win.React.Component = React.Component;
  win.React.Fragment = React.Fragment;
  win.React.StrictMode = React.StrictMode;
  
  // Copy all React exports to window.React
  try {
    Object.getOwnPropertyNames(React).forEach(name => {
      if (!win.React[name]) {
        try {
          win.React[name] = (React as any)[name];
        } catch (e) {
          // Skip non-configurable properties
        }
      }
    });
  } catch (e) {
    console.warn('Could not copy all React properties:', e);
  }
  
  // Verify React.Children exists
  if (!win.React.Children) {
    console.error('React.Children is not available!');
  }
  
  // Mark React as ready and resolve any pending promises
  win.__REACT_READY__ = true;
  if (typeof win.__RESOLVE_REACT_READY__ === 'function') {
    win.__RESOLVE_REACT_READY__();
    delete win.__RESOLVE_REACT_READY__;
  }
  
  // Force a synchronous check
  if (typeof win.React.Children === 'undefined') {
    console.error('CRITICAL: React.Children is still undefined after setup!');
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
