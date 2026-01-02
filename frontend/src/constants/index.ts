/**
 * Application constants
 * Centralized location for all magic numbers and configuration values
 */

export const REFRESH_INTERVALS = {
  API_HEALTH: 60000,        // 1 minute
  GAS_DATA: 30000,          // 30 seconds
  PREDICTIONS: 60000,       // 1 minute
  HISTORICAL: 300000,       // 5 minutes
  USER_HISTORY: 60000,      // 1 minute
  LEADERBOARD: 300000       // 5 minutes
} as const;

export const CACHE_DURATION = {
  SHORT: 10000,             // 10 seconds
  MEDIUM: 60000,            // 1 minute
  LONG: 300000,             // 5 minutes
  VERY_LONG: 3600000        // 1 hour
} as const;

export const RETRY_CONFIG = {
  MAX_ATTEMPTS: 3,
  BASE_DELAY: 1000,         // 1 second
  MAX_DELAY: 10000,         // 10 seconds
  BACKOFF_MULTIPLIER: 2
} as const;

export const RATE_LIMIT = {
  MAX_REQUESTS_PER_MINUTE: 60,
  MAX_REQUESTS_PER_HOUR: 1000,
  WINDOW_MS: 60000          // 1 minute window
} as const;

export const GAS_PRICE_THRESHOLDS = {
  EXCELLENT_RATIO: 0.7,
  GOOD_RATIO: 0.9,
  AVERAGE_RATIO: 1.15,
  HIGH_RATIO: 1.5
} as const;

export const ETH_PRICE = {
  DEFAULT: 3000,
  UPDATE_INTERVAL: 3600000  // 1 hour
} as const;

export const NETWORK = {
  BASE_CHAIN_ID: 8453,
  BLOCK_TIME_SECONDS: 2,
  BLOCKS_PER_HOUR: 1800
} as const;

export const UI = {
  DEBOUNCE_DELAY: 300,      // 300ms
  TOAST_DURATION: 4000,     // 4 seconds
  SKELETON_ANIMATION_DURATION: 1500
} as const;

export const FEATURE_FLAGS = {
  WEBSOCKET_ENABLED: false,
  ANALYTICS_ENABLED: false,
  OFFLINE_MODE: true,
  OPTIMISTIC_UPDATES: true
} as const;
