/**
 * Centralized API configuration
 * All API endpoints and configuration in one place
 */

export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'https://basegasfeesml.onrender.com/api',
  ENDPOINTS: {
    HEALTH: '/health',
    CURRENT: '/current',
    PREDICTIONS: '/predictions',
    HISTORICAL: '/historical',
    TRANSACTIONS: '/transactions',
    CONFIG: '/config',
    ACCURACY: '/accuracy',
    USER_HISTORY: '/user-history',
    LEADERBOARD: '/leaderboard',
    STATS: '/stats'
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
