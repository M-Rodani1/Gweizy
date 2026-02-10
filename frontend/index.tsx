import React from 'react';
import './src/index.css';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';
import { mountApp } from './src/utils/progressiveHydration';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const app = (
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);

mountApp(rootElement, app, { deferHydration: true });
