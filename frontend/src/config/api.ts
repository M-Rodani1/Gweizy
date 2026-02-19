/**
 * Centralized API configuration
 * All API endpoints and configuration in one place
 */

const DEFAULT_API_BASE_URL = '/api';
const DEFAULT_WS_ORIGIN = 'https://basegasfeesml-production.up.railway.app';

function normalizeApiBaseUrl(url: string): string {
  const trimmed = url.replace(/\/+$/, '');
  return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
}

function normalizeOrigin(url: string): string {
  return url.replace(/\/+$/, '').replace(/\/api$/, '');
}

export const API_CONFIG = {
  BASE_URL: normalizeApiBaseUrl(import.meta.env.VITE_API_URL || DEFAULT_API_BASE_URL),
  ENDPOINTS: {
    HEALTH: '/health',
    CURRENT: '/current',
    PREDICTIONS: '/predictions',
    HISTORICAL: '/historical',
    GAS_PATTERNS: '/gas/patterns',
    TRANSACTIONS: '/transactions',
    CONFIG: '/config',
    ACCURACY: '/accuracy',
    USER_HISTORY: '/user-history',
    LEADERBOARD: '/leaderboard',
    STATS: '/stats',
    ETH_PRICE: '/eth-price',
    AGENT_RECOMMEND: '/agent/recommend',
    AGENT_STATUS: '/agent/status',
    VALIDATION_METRICS: '/validation/metrics',
    VALIDATION_HEALTH: '/validation/health',
    VALIDATION_TRENDS: '/validation/trends',
    // Retraining endpoints removed - training done via Colab notebook
    ONCHAIN_NETWORK_STATE: '/onchain/network-state',
    ONCHAIN_CONGESTION_HISTORY: '/onchain/congestion-history',
    ALERTS: '/alerts',
    // Accuracy tracking endpoints
    ACCURACY_METRICS: '/accuracy/metrics',
    ACCURACY_DRIFT: '/accuracy/drift',
    ACCURACY_SUMMARY: '/accuracy/summary',
    ACCURACY_FEATURES: '/accuracy/features',
    ACCURACY_HISTORY: '/accuracy/history',
    ACCURACY_RESET: '/accuracy/reset',
    // Explanation endpoint
    EXPLAIN: '/explain',
    // Data export endpoint
    EXPORT: '/export',
    // Pattern matching endpoint
    PATTERNS: '/patterns',
    // Mempool endpoints
    MEMPOOL_STATUS: '/mempool/status',
    MEMPOOL_HISTORY: '/mempool/history',
    // Advanced analytics endpoints
    ANALYTICS_VOLATILITY: '/analytics/volatility',
    ANALYTICS_WHALES: '/analytics/whales',
    ANALYTICS_ANOMALIES: '/analytics/anomalies',
    ANALYTICS_ENSEMBLE: '/analytics/ensemble',
    // Hybrid prediction endpoint
    HYBRID_PREDICTION: '/predictions/hybrid',
    // Analytics endpoint (for accuracy dashboard)
    ANALYTICS: '/analytics'
  },
  TIMEOUT: 10000, // 10 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000 // 1 second base delay
} as const;

/**
 * Get full API URL for an endpoint
 */
export function getApiUrl(endpoint: string, params?: Record<string, string | number>): string {
  const url = `${API_CONFIG.BASE_URL}${endpoint}`;
  
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      searchParams.append(key, String(value));
    });
    return `${url}?${searchParams.toString()}`;
  }
  
  return url;
}

export function getApiOrigin(): string {
  return API_CONFIG.BASE_URL.replace(/\/api$/, '');
}

export function getWebSocketOrigin(): string {
  const configuredWs = import.meta.env.VITE_WS_URL;
  if (configuredWs && configuredWs.trim().length > 0) {
    return normalizeOrigin(configuredWs.trim());
  }

  const apiOrigin = getApiOrigin();
  if (!apiOrigin || apiOrigin.startsWith('/')) {
    return DEFAULT_WS_ORIGIN;
  }

  return normalizeOrigin(apiOrigin);
}

/**
 * Base RPC endpoints for Base network
 */
export const BASE_RPC_CONFIG = {
  ENDPOINTS: [
    'https://mainnet.base.org',
    'https://base.llamarpc.com',
    'https://base-rpc.publicnode.com'
  ],
  TIMEOUT: 5000,
  RETRY_ATTEMPTS: 3
} as const;
