/**
 * Tests for gasApi.ts
 *
 * Tests cover:
 * - Retry logic with exponential backoff
 * - Caching behavior (read/write/expiry)
 * - Error handling (network, timeout, HTTP errors)
 * - Request deduplication
 * - All API functions
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock the config module
vi.mock('../../config/api', () => ({
  API_CONFIG: {
    BASE_URL: 'https://api.test.com/api',
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
      STATS: '/stats',
      HYBRID_PREDICTION: '/predictions/hybrid',
    },
    TIMEOUT: 10000,
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000,
  },
  getApiUrl: vi.fn((endpoint: string, params?: Record<string, string | number>) => {
    const url = `https://api.test.com/api${endpoint}`;
    if (params) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        searchParams.append(key, String(value));
      });
      return `${url}?${searchParams.toString()}`;
    }
    return url;
  }),
}));

// Mock request deduplicator
vi.mock('../../utils/requestDeduplication', () => ({
  requestDeduplicator: {
    deduplicate: vi.fn((_key: string, fn: () => Promise<unknown>) => fn()),
  },
}));

// Import after mocks
import {
  checkHealth,
  fetchCurrentGas,
  fetchPredictions,
  fetchHistoricalData,
  fetchTransactions,
  fetchConfig,
  fetchAccuracy,
  fetchUserHistory,
  fetchLeaderboard,
  fetchGlobalStats,
  fetchHybridPrediction,
  GasAPIError,
  __resetGasApiCircuitBreakerForTests,
} from '../../api/gasApi';
import { requestDeduplicator } from '../../utils/requestDeduplication';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(global, 'localStorage', {
  value: localStorageMock,
  writable: true,
});

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock AbortController
class MockAbortController {
  signal = { aborted: false };
  abort = vi.fn(() => {
    this.signal.aborted = true;
  });
}
global.AbortController = MockAbortController as unknown as typeof AbortController;

describe('gasApi', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorageMock.clear();
    vi.useFakeTimers();
    __resetGasApiCircuitBreakerForTests();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('GasAPIError', () => {
    it('should create error with message and status', () => {
      const error = new GasAPIError('Test error', 500);
      expect(error.message).toBe('Test error');
      expect(error.status).toBe(500);
      expect(error.name).toBe('GasAPIError');
    });

    it('should create error without status', () => {
      const error = new GasAPIError('Test error');
      expect(error.message).toBe('Test error');
      expect(error.status).toBeUndefined();
    });
  });

  describe('checkHealth', () => {
    it('should return true when API is healthy', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
      });

      const result = await checkHealth();
      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/api/health',
        expect.objectContaining({
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });

    it('should return false when API returns error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      });

      const result = await checkHealth();
      expect(result).toBe(false);
    });

    it('should return false when fetch throws', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const result = await checkHealth();
      expect(result).toBe(false);
    });
  });

  describe('fetchCurrentGas', () => {
    const mockGasData = {
      timestamp: '2024-01-01T00:00:00Z',
      current_gas: 0.001,
      base_fee: 0.0008,
      priority_fee: 0.0002,
      block_number: 12345678,
    };

    it('should fetch current gas successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockGasData),
      });

      const result = await fetchCurrentGas();
      expect(result).toEqual(mockGasData);
    });

    it('should cache successful response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockGasData),
      });

      await fetchCurrentGas();

      expect(localStorageMock.setItem).toHaveBeenCalled();
      const cachedData = JSON.parse(localStorageMock.setItem.mock.calls[0][1]);
      expect(cachedData.data).toEqual(mockGasData);
    });

    it('should use request deduplication', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockGasData),
      });

      await fetchCurrentGas();

      expect(requestDeduplicator.deduplicate).toHaveBeenCalledWith(
        'GET:https://api.test.com/api/current',
        expect.any(Function)
      );
    });
  });

  describe('fetchPredictions', () => {
    const mockPredictions = {
      current: { current_gas: 0.001 },
      predictions: {
        '1h': [{ predictedGwei: 0.0012 }],
        '4h': [{ predictedGwei: 0.0015 }],
        '24h': [{ predictedGwei: 0.002 }],
      },
    };

    it('should fetch predictions without chainId', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPredictions),
      });

      const result = await fetchPredictions();
      expect(result).toEqual(mockPredictions);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/api/predictions',
        expect.any(Object)
      );
    });

    it('should fetch predictions with chainId', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPredictions),
      });

      await fetchPredictions(8453);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/api/predictions?chain_id=8453',
        expect.any(Object)
      );
    });
  });

  describe('fetchHistoricalData', () => {
    const mockHistorical = {
      data: [{ timestamp: '2024-01-01', gwei: 0.001 }],
    };

    it('should fetch historical data with default hours', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHistorical),
      });

      await fetchHistoricalData();
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/api/historical?hours=168',
        expect.any(Object)
      );
    });

    it('should fetch historical data with custom hours', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHistorical),
      });

      await fetchHistoricalData(24);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/api/historical?hours=24',
        expect.any(Object)
      );
    });
  });

  describe('fetchTransactions', () => {
    const mockTransactions = {
      transactions: [
        { hash: '0x123', gas_price: 0.001 },
        { hash: '0x456', gas_price: 0.002 },
      ],
    };

    it('should return transactions array', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockTransactions),
      });

      const result = await fetchTransactions();
      expect(result).toEqual(mockTransactions.transactions);
    });

    it('should return empty array when no transactions', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({}),
      });

      const result = await fetchTransactions();
      expect(result).toEqual([]);
    });

    it('should pass limit parameter', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockTransactions),
      });

      await fetchTransactions(20);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/api/transactions?limit=20',
        expect.any(Object)
      );
    });
  });

  describe('fetchConfig', () => {
    const mockConfig = {
      chains: [{ id: 8453, name: 'Base', enabled: true }],
      features: { predictions: true },
      version: '1.0.0',
    };

    it('should fetch config successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockConfig),
      });

      const result = await fetchConfig();
      expect(result).toEqual(mockConfig);
    });
  });

  describe('fetchAccuracy', () => {
    const mockAccuracy = {
      horizons: { '1h': { mae: 0.001, n_samples: 100 } },
      overall: { mae: 0.002 },
      updated_at: '2024-01-01T00:00:00Z',
    };

    it('should fetch accuracy successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockAccuracy),
      });

      const result = await fetchAccuracy();
      expect(result).toEqual(mockAccuracy);
    });
  });

  describe('fetchUserHistory', () => {
    const mockHistory = {
      address: '0x123',
      transactions: [],
      total_savings: 100,
      total_transactions: 10,
    };

    it('should fetch user history with address', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHistory),
      });

      const result = await fetchUserHistory('0x123');
      expect(result).toEqual(mockHistory);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/api/user-history/0x123',
        expect.any(Object)
      );
    });
  });

  describe('fetchLeaderboard', () => {
    const mockLeaderboard = {
      entries: [{ rank: 1, address: '0x123', total_savings: 1000 }],
      updated_at: '2024-01-01T00:00:00Z',
    };

    it('should fetch leaderboard successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockLeaderboard),
      });

      const result = await fetchLeaderboard();
      expect(result).toEqual(mockLeaderboard);
    });
  });

  describe('fetchGlobalStats', () => {
    const mockStats = {
      total_users: 1000,
      total_transactions: 50000,
      total_savings_usd: 100000,
      predictions_made: 500000,
      average_accuracy: 0.95,
      active_chains: 3,
    };

    it('should fetch global stats successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockStats),
      });

      const result = await fetchGlobalStats();
      expect(result).toEqual(mockStats);
    });
  });

  describe('fetchHybridPrediction', () => {
    const mockHybrid = {
      action: 'WAIT',
      confidence: 0.85,
      trend_signal_4h: -0.2,
      probabilities: { wait: 0.6, normal: 0.3, urgent: 0.1 },
    };

    it('should fetch hybrid prediction successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHybrid),
      });

      const result = await fetchHybridPrediction();
      expect(result).toEqual(mockHybrid);
    });
  });

  describe('Retry logic', () => {
    it('should retry on 5xx errors', async () => {
      const mockData = { current_gas: 0.001 };

      // First two attempts fail with 500, third succeeds
      mockFetch
        .mockResolvedValueOnce({ ok: false, status: 500, statusText: 'Server Error' })
        .mockResolvedValueOnce({ ok: false, status: 503, statusText: 'Service Unavailable' })
        .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockData) });

      const resultPromise = fetchCurrentGas();

      // Advance timers for retry delays
      await vi.advanceTimersByTimeAsync(1000); // First retry delay
      await vi.advanceTimersByTimeAsync(2000); // Second retry delay

      const result = await resultPromise;
      expect(result).toEqual(mockData);
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });

    it('should not retry on 4xx errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      });

      await expect(fetchCurrentGas()).rejects.toThrow(GasAPIError);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should retry on network errors', async () => {
      const mockData = { current_gas: 0.001 };

      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Connection reset'))
        .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(mockData) });

      const resultPromise = fetchCurrentGas();

      await vi.advanceTimersByTimeAsync(1000);
      await vi.advanceTimersByTimeAsync(2000);

      const result = await resultPromise;
      expect(result).toEqual(mockData);
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });
  });

  describe('Cache behavior', () => {
    it('should return cached data when all retries fail', async () => {
      const cachedData = { current_gas: 0.001 };

      // Pre-populate cache
      localStorageMock.getItem.mockReturnValueOnce(
        JSON.stringify({ timestamp: Date.now(), data: cachedData })
      );

      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'));

      const resultPromise = fetchCurrentGas();

      await vi.advanceTimersByTimeAsync(1000);
      await vi.advanceTimersByTimeAsync(2000);

      const result = await resultPromise;
      expect(result).toEqual(cachedData);
    });

    it('should ignore expired cache', async () => {
      const cachedData = { current_gas: 0.001 };

      // Pre-populate cache with expired data
      localStorageMock.getItem.mockReturnValueOnce(
        JSON.stringify({ timestamp: Date.now() - 100000, data: cachedData })
      );

      mockFetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Network error'));

      const resultPromise = fetchCurrentGas();
      const rejection = expect(resultPromise).rejects.toThrow(GasAPIError);

      await vi.advanceTimersByTimeAsync(1000);
      await vi.advanceTimersByTimeAsync(2000);

      await rejection;
    });
  });

  describe('Error handling', () => {
    it('should throw GasAPIError on HTTP error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
      });

      // 400 errors don't retry (< 500), so should fail immediately
      await expect(fetchCurrentGas()).rejects.toThrow(GasAPIError);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should throw GasAPIError on timeout', async () => {
      // Mock AbortError
      const abortError = new Error('Aborted');
      abortError.name = 'AbortError';

      mockFetch.mockRejectedValue(abortError);

      const resultPromise = fetchCurrentGas();
      const rejection = expect(resultPromise).rejects.toThrow('request timed out');

      // Advance through all retry delays
      await vi.advanceTimersByTimeAsync(1000);
      await vi.advanceTimersByTimeAsync(2000);
      await vi.advanceTimersByTimeAsync(3000);

      await rejection;
    });

    it('should include status code in error', async () => {
      // 503 errors will retry, need to make all retries fail
      mockFetch.mockResolvedValue({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
      });

      const resultPromise = fetchCurrentGas();
      const rejection = expect(resultPromise).rejects.toMatchObject({ status: 503 });

      // Advance through retry delays
      await vi.advanceTimersByTimeAsync(1000);
      await vi.advanceTimersByTimeAsync(2000);
      await vi.advanceTimersByTimeAsync(3000);

      await rejection;
    });
  });

  describe('Circuit breaker', () => {
    const advanceRetryDelays = async () => {
      await vi.advanceTimersByTimeAsync(1000);
      await vi.advanceTimersByTimeAsync(2000);
    };

    it('opens after repeated retryable failures and short-circuits subsequent requests', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
      });

      for (let i = 0; i < 3; i += 1) {
        const failingRequest = fetchCurrentGas();
        const failingExpectation = expect(failingRequest).rejects.toMatchObject({ status: 503 });
        await advanceRetryDelays();
        await failingExpectation;
      }

      const callsBeforeShortCircuit = mockFetch.mock.calls.length;
      await expect(fetchCurrentGas()).rejects.toMatchObject({ status: 503 });
      expect(mockFetch).toHaveBeenCalledTimes(callsBeforeShortCircuit);
    });

    it('returns cached data while circuit is open', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
      });

      for (let i = 0; i < 3; i += 1) {
        const failingRequest = fetchCurrentGas();
        const failingExpectation = expect(failingRequest).rejects.toMatchObject({ status: 503 });
        await advanceRetryDelays();
        await failingExpectation;
      }

      const callsBeforeCachedShortCircuit = mockFetch.mock.calls.length;
      const cachedData = { current_gas: 0.001, source: 'cache' };
      localStorageMock.getItem.mockReturnValueOnce(
        JSON.stringify({ timestamp: Date.now(), data: cachedData })
      );

      const result = await fetchCurrentGas();
      expect(result).toEqual(cachedData);
      expect(mockFetch).toHaveBeenCalledTimes(callsBeforeCachedShortCircuit);
    });

    it('closes after cooldown and allows live requests again', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
      });

      for (let i = 0; i < 3; i += 1) {
        const failingRequest = fetchCurrentGas();
        const failingExpectation = expect(failingRequest).rejects.toMatchObject({ status: 503 });
        await advanceRetryDelays();
        await failingExpectation;
      }

      const callsBeforeCooldownRecovery = mockFetch.mock.calls.length;
      mockFetch.mockReset();
      const liveData = { current_gas: 0.002 };
      mockFetch.mockResolvedValueOnce({ ok: true, json: () => Promise.resolve(liveData) });

      await vi.advanceTimersByTimeAsync(60001);
      const result = await fetchCurrentGas();

      expect(result).toEqual(liveData);
      expect(mockFetch).toHaveBeenCalledTimes(1);
      expect(callsBeforeCooldownRecovery).toBeGreaterThan(0);
    });
  });
});
