import React from 'react';
import ReactDOM from 'react-dom/client';
import App from '../App';
import ErrorBoundary from '../components/ErrorBoundary';
import './index.css';

const root = document.getElementById('root');
if (root) {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </React.StrictMode>
  );
}
// Cache bust Thu Dec 18 22:30:54 CET 2025
