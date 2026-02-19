import { API_CONFIG, getApiUrl } from '../config/api';
import { requestDeduplicator } from '../utils/requestDeduplication';

import {
  CurrentGasData,
  PredictionsResponse,
  TableRowData,
  HistoricalResponse,
  HybridPrediction,
  ConfigResponse,
  AccuracyResponse,
  UserHistoryResponse,
  LeaderboardResponse,
  GlobalStatsResponse
} from '../../types';

class GasAPIError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'GasAPIError';
  }
}

const CACHE_PREFIX = 'gweizy_api_cache_v1:';
const ERROR_LOG_THROTTLE_MS = 30000;
const CACHE_TTL_MS = {
  current: 15000,
  predictions: 60000,
  historical: 10 * 60 * 1000,
  transactions: 30000,
  config: 5 * 60 * 1000,
  accuracy: 10 * 60 * 1000,
  userHistory: 60 * 1000,
  leaderboard: 2 * 60 * 1000,
  stats: 60 * 1000
};

const getCacheKey = (url: string): string => `${CACHE_PREFIX}${url}`;
const lastErrorLoggedAt = new Map<string, number>();

const logApiErrorThrottled = (key: string, ...args: unknown[]) => {
  const now = Date.now();
  const lastLoggedAt = lastErrorLoggedAt.get(key) || 0;
  if (now - lastLoggedAt < ERROR_LOG_THROTTLE_MS) {
    return;
  }
  lastErrorLoggedAt.set(key, now);
  console.error(...args);
};

const readCache = <T,>(key: string, ttlMs: number): T | null => {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { timestamp: number; data: T };
    if (Date.now() - parsed.timestamp > ttlMs) {
      localStorage.removeItem(key);
      return null;
    }
    return parsed.data;
  } catch {
    return null;
  }
};

const writeCache = (key: string, data: unknown) => {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(key, JSON.stringify({ timestamp: Date.now(), data }));
  } catch {
    // Ignore cache write failures.
  }
};

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const fetchJsonWithRetry = async <T,>(
  url: string,
  options: RequestInit,
  cacheConfig: { key: string; ttlMs: number } | null,
  errorMessage: string
): Promise<T> => {
  const dedupeKey = `${options.method || 'GET'}:${url}`;

  return requestDeduplicator.deduplicate(dedupeKey, async () => {
    let lastError: unknown;

    for (let attempt = 0; attempt < API_CONFIG.RETRY_ATTEMPTS; attempt += 1) {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.TIMEOUT);
      try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        clearTimeout(timeoutId);

        if (!response.ok) {
          const apiError = new GasAPIError(`${errorMessage}: ${response.statusText}`, response.status);
          lastError = apiError;
          if (attempt < API_CONFIG.RETRY_ATTEMPTS - 1 && response.status >= 500) {
            await sleep(API_CONFIG.RETRY_DELAY * (attempt + 1));
            continue;
          }
          break;
        }

        const data = await response.json();
        if (cacheConfig) {
          writeCache(cacheConfig.key, data);
        }
        return data as T;
      } catch (err) {
        clearTimeout(timeoutId);
        lastError = err;
        if (attempt < API_CONFIG.RETRY_ATTEMPTS - 1) {
          await sleep(API_CONFIG.RETRY_DELAY * (attempt + 1));
          continue;
        }
      }
    }

    if (cacheConfig) {
      const cached = readCache<T>(cacheConfig.key, cacheConfig.ttlMs);
      if (cached) {
        return cached;
      }
    }

    if (lastError instanceof GasAPIError) {
      logApiErrorThrottled(
        `api:${url}:${errorMessage}:http:${lastError.status || 'unknown'}`,
        '[GasAPI]',
        lastError.message
      );
      throw lastError;
    }
    const isAbortError =
      (typeof DOMException !== 'undefined' && lastError instanceof DOMException && lastError.name === 'AbortError') ||
      (typeof lastError === 'object' && lastError !== null && 'name' in lastError && (lastError as { name?: string }).name === 'AbortError');
    if (isAbortError) {
      logApiErrorThrottled(
        `api:${url}:${errorMessage}:timeout`,
        '[GasAPI]',
        `${errorMessage}: request timed out`
      );
      throw new GasAPIError(`${errorMessage}: request timed out`);
    }
    logApiErrorThrottled(
      `api:${url}:${errorMessage}:network`,
      '[GasAPI]',
      errorMessage,
      lastError
    );
    throw new GasAPIError(errorMessage);
  });
};

/**
 * Check if API is healthy
 */
export async function checkHealth(): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.TIMEOUT);
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.HEALTH), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      mode: 'cors',
      credentials: 'omit',
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    if (!response.ok) {
      logApiErrorThrottled(
        `health:http:${response.status}`,
        'Health check failed: HTTP',
        response.status,
        response.statusText
      );
      return false;
    }
    return true;
  } catch (error) {
    clearTimeout(timeoutId);
    logApiErrorThrottled('health:network', 'Health check failed:', error);
    logApiErrorThrottled('health:base-url', 'API_BASE_URL:', API_CONFIG.BASE_URL);
    return false;
  }
}

/**
 * Fetch current Base gas price
 */
export async function fetchCurrentGas(): Promise<CurrentGasData> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.CURRENT);
  return fetchJsonWithRetry<CurrentGasData>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.current },
    'Failed to fetch current gas'
  );
}

/**
 * Fetch ML predictions and historical data for a specific chain
 * @param chainId - Chain ID (defaults to Base/8453 if not provided)
 */
export async function fetchPredictions(chainId?: number): Promise<PredictionsResponse> {
  const params = chainId ? { chain_id: chainId.toString() } : undefined;
  const url = getApiUrl(API_CONFIG.ENDPOINTS.PREDICTIONS, params);
  return fetchJsonWithRetry<PredictionsResponse>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.predictions },
    'Failed to fetch predictions'
  );
}

/**
 * Fetch historical gas prices
 */
export async function fetchHistoricalData(hours: number = 168): Promise<HistoricalResponse> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.HISTORICAL, { hours });
  return fetchJsonWithRetry<HistoricalResponse>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.historical },
    'Failed to fetch historical data'
  );
}

/**
 * Fetch recent Base transactions
 */
export async function fetchTransactions(limit: number = 10): Promise<TableRowData[]> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.TRANSACTIONS, { limit });
  const data = await fetchJsonWithRetry<{ transactions?: TableRowData[] }>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.transactions },
    'Failed to fetch transactions'
  );
  return data.transactions || [];
}

/**
 * Fetch Base platform config
 */
export async function fetchConfig(): Promise<ConfigResponse> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.CONFIG);
  return fetchJsonWithRetry<ConfigResponse>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.config },
    'Failed to fetch config'
  );
}

/**
 * Fetch model accuracy metrics
 */
export async function fetchAccuracy(): Promise<AccuracyResponse> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY);
  return fetchJsonWithRetry<AccuracyResponse>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.accuracy },
    'Failed to fetch accuracy'
  );
}

/**
 * Fetch user transaction history
 */
export async function fetchUserHistory(address: string): Promise<UserHistoryResponse> {
  const url = getApiUrl(`${API_CONFIG.ENDPOINTS.USER_HISTORY}/${address}`);
  return fetchJsonWithRetry<UserHistoryResponse>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.userHistory },
    'Failed to fetch user history'
  );
}

/**
 * Fetch savings leaderboard
 */
export async function fetchLeaderboard(): Promise<LeaderboardResponse> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.LEADERBOARD);
  return fetchJsonWithRetry<LeaderboardResponse>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.leaderboard },
    'Failed to fetch leaderboard'
  );
}

/**
 * Fetch global statistics for landing page - LIVE DATA
 */
export async function fetchGlobalStats(): Promise<GlobalStatsResponse> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.STATS);
  return fetchJsonWithRetry<GlobalStatsResponse>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.stats },
    'Failed to fetch stats'
  );
}

/**
 * Fetch hybrid model prediction
 * Returns action recommendation (WAIT/NORMAL/URGENT) with probabilities and trend signal
 */
export async function fetchHybridPrediction(): Promise<HybridPrediction> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.HYBRID_PREDICTION);
  return fetchJsonWithRetry<HybridPrediction>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.predictions },
    'Failed to fetch hybrid prediction'
  );
}

/**
 * Fetch analytics performance metrics.
 */
export async function fetchAnalyticsPerformance(days: number = 90): Promise<Record<string, unknown>> {
  const url = getApiUrl(`${API_CONFIG.ENDPOINTS.ANALYTICS}/performance`, { days });
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.accuracy },
    'Failed to fetch analytics performance'
  );
}

/**
 * Fetch analytics trend data for a horizon.
 */
export async function fetchAnalyticsTrends(
  horizon: '1h' | '4h' | '24h',
  days: number = 90
): Promise<Record<string, unknown>> {
  const url = getApiUrl(`${API_CONFIG.ENDPOINTS.ANALYTICS}/trends`, { horizon, days });
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.accuracy },
    'Failed to fetch analytics trends'
  );
}

/**
 * Fetch on-chain network state.
 */
export async function fetchOnchainNetworkState(): Promise<Record<string, unknown>> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.ONCHAIN_NETWORK_STATE);
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.current },
    'Failed to fetch on-chain network state'
  );
}

/**
 * Fetch on-chain congestion history.
 */
export async function fetchOnchainCongestionHistory(hours: number = 24): Promise<Record<string, unknown>> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.ONCHAIN_CONGESTION_HISTORY, { hours });
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.historical },
    'Failed to fetch on-chain congestion history'
  );
}

/**
 * Fetch agent status.
 */
export async function fetchAgentStatus(): Promise<Record<string, unknown>> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.AGENT_STATUS);
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.current },
    'Failed to fetch agent status'
  );
}

/**
 * Fetch drift metrics from accuracy tracking.
 */
export async function fetchAccuracyDrift(): Promise<Record<string, unknown>> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_DRIFT);
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.accuracy },
    'Failed to fetch accuracy drift'
  );
}

/**
 * Fetch model accuracy metrics payload.
 */
export async function fetchAccuracyMetrics(): Promise<Record<string, unknown>> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_METRICS);
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.accuracy },
    'Failed to fetch accuracy metrics'
  );
}

/**
 * Fetch validation trends.
 */
export async function fetchValidationTrends(
  horizon: '1h' | '4h' | '24h',
  days: number
): Promise<Record<string, unknown>> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.VALIDATION_TRENDS, { horizon, days });
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.accuracy },
    'Failed to fetch validation trends'
  );
}

/**
 * Fetch gas pattern data.
 */
export async function fetchGasPatterns(): Promise<Record<string, unknown>> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.GAS_PATTERNS);
  return fetchJsonWithRetry<Record<string, unknown>>(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.historical },
    'Failed to fetch gas patterns'
  );
}

// Export error class
export { GasAPIError };
