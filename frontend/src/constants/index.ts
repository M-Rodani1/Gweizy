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

// ========================================
// Chain Configuration
// ========================================

export const CHAIN_IDS = {
  ETHEREUM: 1,
  BASE: 8453,
  ARBITRUM: 42161,
  OPTIMISM: 10,
  POLYGON: 137,
} as const;

export const CHAIN_NAMES: Record<number, string> = {
  [CHAIN_IDS.ETHEREUM]: 'Ethereum',
  [CHAIN_IDS.BASE]: 'Base',
  [CHAIN_IDS.ARBITRUM]: 'Arbitrum',
  [CHAIN_IDS.OPTIMISM]: 'Optimism',
  [CHAIN_IDS.POLYGON]: 'Polygon',
};

// ========================================
// Confidence Levels
// ========================================

export const CONFIDENCE_LEVELS = {
  LOW: 0.5,
  MEDIUM: 0.7,
  HIGH: 0.85,
  VERY_HIGH: 0.95,
} as const;

// ========================================
// Pagination
// ========================================

export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 10,
  MAX_PAGE_SIZE: 100,
  PAGE_SIZE_OPTIONS: [10, 25, 50, 100] as const,
} as const;

// ========================================
// Local Storage Keys
// ========================================

export const STORAGE_KEYS = {
  THEME: 'gweizy-theme',
  CHAIN_PREFERENCE: 'gweizy-chain',
  GAS_CACHE: 'gweizy-gas-cache',
  PREDICTIONS_CACHE: 'gweizy-predictions-cache',
  USER_PREFERENCES: 'gweizy-preferences',
  SCHEDULED_TXS: 'gweizy-scheduled-txs',
} as const;

// ========================================
// Time Constants
// ========================================

export const TIME = {
  SECOND: 1000,
  MINUTE: 60 * 1000,
  HOUR: 60 * 60 * 1000,
  DAY: 24 * 60 * 60 * 1000,
} as const;

// ========================================
// Validation
// ========================================

export const VALIDATION = {
  MIN_ADDRESS_LENGTH: 42,
  MAX_HASH_LENGTH: 66,
  MAX_SEARCH_LENGTH: 200,
} as const;

// ========================================
// WebSocket Configuration
// ========================================

export const WEBSOCKET_CONFIG = {
  MAX_RECONNECT_ATTEMPTS: 5,
  INITIAL_RECONNECT_DELAY: 1000,
  MAX_RECONNECT_DELAY: 5000,
  CONNECTION_TIMEOUT: 20000,
} as const;

// ========================================
// Virtual Scrolling
// ========================================

export const VIRTUAL_SCROLL = {
  THRESHOLD: 50,
  ROW_HEIGHT: 60,
  OVERSCAN: 5,
} as const;
