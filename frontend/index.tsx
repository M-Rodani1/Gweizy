import React from 'react';
import './src/index.css';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';
import { mountApp } from './src/utils/progressiveHydration';
import { applyResourceHints, dnsPrefetch, preconnect } from './src/utils/resourceHints';
import { BASE_RPC_CONFIG, getApiOrigin } from './src/config/api';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const apiOrigin = getApiOrigin();
const rpcHints = BASE_RPC_CONFIG.ENDPOINTS.flatMap((endpoint) => [
  preconnect(endpoint),
  dnsPrefetch(endpoint),
]);

applyResourceHints([preconnect(apiOrigin), dnsPrefetch(apiOrigin), ...rpcHints]);

const app = (
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);

mountApp(rootElement, app, { deferHydration: true });
