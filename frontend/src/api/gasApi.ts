import { API_CONFIG, getApiUrl } from '../config/api';

import {
  CurrentGasData,
  PredictionsResponse,
  TableRowData,
  HistoricalResponse,
  APIError
} from '../../types';

class GasAPIError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
    this.name = 'GasAPIError';
  }
}

/**
 * Check if API is healthy
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.HEALTH), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      mode: 'cors',
      credentials: 'omit',
    });
    if (!response.ok) {
      console.error('Health check failed: HTTP', response.status, response.statusText);
      return false;
    }
    return true;
  } catch (error) {
    console.error('Health check failed:', error);
    console.error('API_BASE_URL:', API_CONFIG.BASE_URL);
    return false;
  }
}

/**
 * Fetch current Base gas price
 */
export async function fetchCurrentGas(): Promise<CurrentGasData> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.CURRENT), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch current gas: ${response.statusText}`, response.status);
    }

    const data: CurrentGasData = await response.json();
    return data;
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching current gas');
  }
}

/**
 * Fetch ML predictions and historical data
 */
export async function fetchPredictions(): Promise<PredictionsResponse> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.PREDICTIONS), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch predictions: ${response.statusText}`, response.status);
    }

    const data: PredictionsResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching predictions');
  }
}

/**
 * Fetch historical gas prices
 */
export async function fetchHistoricalData(hours: number = 168): Promise<HistoricalResponse> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.HISTORICAL, { hours }), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch historical data: ${response.statusText}`, response.status);
    }

    const data: HistoricalResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching historical data');
  }
}

/**
 * Fetch recent Base transactions
 */
export async function fetchTransactions(limit: number = 10): Promise<TableRowData[]> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.TRANSACTIONS, { limit }), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch transactions: ${response.statusText}`, response.status);
    }

    const data = await response.json();
    return data.transactions || [];
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching transactions');
  }
}

/**
 * Fetch Base platform config
 */
export async function fetchConfig(): Promise<any> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.CONFIG), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch config: ${response.statusText}`, response.status);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching config');
  }
}

/**
 * Fetch model accuracy metrics
 */
export async function fetchAccuracy(): Promise<any> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch accuracy: ${response.statusText}`, response.status);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching accuracy');
  }
}

/**
 * Fetch user transaction history
 */
export async function fetchUserHistory(address: string): Promise<any> {
  try {
    const response = await fetch(getApiUrl(`${API_CONFIG.ENDPOINTS.USER_HISTORY}/${address}`), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch user history: ${response.statusText}`, response.status);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching user history');
  }
}

/**
 * Fetch savings leaderboard
 */
export async function fetchLeaderboard(): Promise<any> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.LEADERBOARD), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch leaderboard: ${response.statusText}`, response.status);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching leaderboard');
  }
}

/**
 * Fetch global statistics for landing page - LIVE DATA
 */
export async function fetchGlobalStats(): Promise<any> {
  try {
    const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.STATS), {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      throw new GasAPIError(`Failed to fetch stats: ${response.statusText}`, response.status);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof GasAPIError) throw error;
    throw new GasAPIError('Network error fetching stats');
  }
}

// Export error class
export { GasAPIError };
