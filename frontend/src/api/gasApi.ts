import { API_CONFIG, getApiUrl } from '../config/api';

import {
  CurrentGasData,
  PredictionsResponse,
  TableRowData,
  HistoricalResponse
} from '../../types';

class GasAPIError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'GasAPIError';
  }
}

const CACHE_PREFIX = 'gweizy_api_cache_v1:';
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
    throw lastError;
  }
  const isAbortError =
    (typeof DOMException !== 'undefined' && lastError instanceof DOMException && lastError.name === 'AbortError') ||
    (typeof lastError === 'object' && lastError !== null && 'name' in lastError && (lastError as { name?: string }).name === 'AbortError');
  if (isAbortError) {
    throw new GasAPIError(`${errorMessage}: request timed out`);
  }
  throw new GasAPIError(errorMessage);
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
      console.error('Health check failed: HTTP', response.status, response.statusText);
      return false;
    }
    return true;
  } catch (error) {
    clearTimeout(timeoutId);
    console.error('Health check failed:', error);
    console.error('API_BASE_URL:', API_CONFIG.BASE_URL);
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
 * Fetch ML predictions and historical data
 */
export async function fetchPredictions(): Promise<PredictionsResponse> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.PREDICTIONS);
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
export async function fetchConfig(): Promise<any> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.CONFIG);
  return fetchJsonWithRetry(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.config },
    'Failed to fetch config'
  );
}

/**
 * Fetch model accuracy metrics
 */
export async function fetchAccuracy(): Promise<any> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY);
  return fetchJsonWithRetry(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.accuracy },
    'Failed to fetch accuracy'
  );
}

/**
 * Fetch user transaction history
 */
export async function fetchUserHistory(address: string): Promise<any> {
  const url = getApiUrl(`${API_CONFIG.ENDPOINTS.USER_HISTORY}/${address}`);
  return fetchJsonWithRetry(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.userHistory },
    'Failed to fetch user history'
  );
}

/**
 * Fetch savings leaderboard
 */
export async function fetchLeaderboard(): Promise<any> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.LEADERBOARD);
  return fetchJsonWithRetry(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.leaderboard },
    'Failed to fetch leaderboard'
  );
}

/**
 * Fetch global statistics for landing page - LIVE DATA
 */
export async function fetchGlobalStats(): Promise<any> {
  const url = getApiUrl(API_CONFIG.ENDPOINTS.STATS);
  return fetchJsonWithRetry(
    url,
    { method: 'GET', headers: { 'Content-Type': 'application/json' } },
    { key: getCacheKey(url), ttlMs: CACHE_TTL_MS.stats },
    'Failed to fetch stats'
  );
}

// Export error class
export { GasAPIError };
