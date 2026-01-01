/**
 * Centralized API configuration
 * All API endpoints and configuration in one place
 */

const DEFAULT_API_BASE_URL = 'https://basegasfeesml.onrender.com/api';

function normalizeApiBaseUrl(url: string): string {
  const trimmed = url.replace(/\/+$/, '');
  return trimmed.endsWith('/api') ? trimmed : `${trimmed}/api`;
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
    AGENT_RECOMMEND: '/agent/recommend',
    AGENT_STATUS: '/agent/status',
    VALIDATION_METRICS: '/validation/metrics',
    VALIDATION_HEALTH: '/validation/health',
    VALIDATION_TRENDS: '/validation/trends',
    RETRAINING_STATUS: '/retraining/status',
    RETRAINING_HISTORY: '/retraining/history',
    RETRAINING_TRIGGER: '/retraining/trigger',
    RETRAINING_CHECK_DATA: '/retraining/check-data',
    ONCHAIN_NETWORK_STATE: '/onchain/network-state',
    ONCHAIN_CONGESTION_HISTORY: '/onchain/congestion-history',
    ALERTS: '/alerts',
    // Accuracy tracking endpoints
    ACCURACY_METRICS: '/accuracy/metrics',
    ACCURACY_DRIFT: '/accuracy/drift',
    ACCURACY_SUMMARY: '/accuracy/summary',
    ACCURACY_FEATURES: '/accuracy/features'
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
