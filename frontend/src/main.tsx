import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '../App';
import ErrorBoundary from '../components/ErrorBoundary';
import { QueryProvider } from './providers/QueryProvider';
import { ToastProvider } from './providers/ToastProvider';
import { OfflineIndicator } from './components/OfflineIndicator';
import './index.css';

const root = document.getElementById('root');
if (root) {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <ErrorBoundary>
        <QueryProvider>
          <ToastProvider>
            <OfflineIndicator />
            <App />
          </ToastProvider>
        </QueryProvider>
      </ErrorBoundary>
    </React.StrictMode>
  );
}
// Cache bust Thu Dec 18 22:30:54 CET 2025
