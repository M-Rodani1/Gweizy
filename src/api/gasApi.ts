const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://basegasfeesml.onrender.com/api';

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
    const response = await fetch(`${API_BASE_URL}/health`, {
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
    console.error('API_BASE_URL:', API_BASE_URL);
    return false;
  }
}

/**
 * Fetch current Base gas price
 */
export async function fetchCurrentGas(): Promise<CurrentGasData> {
  try {
    const response = await fetch(`${API_BASE_URL}/current`, {
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
    const response = await fetch(`${API_BASE_URL}/predictions`, {
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
    const response = await fetch(`${API_BASE_URL}/historical?hours=${hours}`, {
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
    const response = await fetch(`${API_BASE_URL}/transactions?limit=${limit}`, {
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
    const response = await fetch(`${API_BASE_URL}/config`, {
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
    const response = await fetch(`${API_BASE_URL}/accuracy`, {
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
    const response = await fetch(`${API_BASE_URL}/user-history/${address}`, {
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
    const response = await fetch(`${API_BASE_URL}/leaderboard`, {
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
 * Fetch global statistics for landing page
 */
export async function fetchGlobalStats(): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/stats`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      console.warn(`Stats API returned ${response.status}, using fallback`);
      return {
        success: true,
        stats: {
          total_saved_k: 52,
          accuracy_percent: 82,
          predictions_k: 15
        }
      };
    }

    return await response.json();
  } catch (error) {
    console.warn('Stats API failed, using fallback:', error);
    return {
      success: true,
      stats: {
        total_saved_k: 52,
        accuracy_percent: 82,
        predictions_k: 15
      }
    };
  }
}

// Export error class
export { GasAPIError };

