/**
 * Integration tests for API calls
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchCurrentGas, fetchPredictions, checkHealth } from '../../api/gasApi';
import { API_CONFIG } from '../../config/api';
import { requestDeduplicator } from '../../utils/requestDeduplication';

// Mock fetch globally
global.fetch = vi.fn();

describe('API Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    requestDeduplicator.clear();
    if (typeof globalThis !== 'undefined' && 'localStorage' in globalThis) {
      try {
        globalThis.localStorage.clear();
      } catch {
        // Ignore storage cleanup errors in tests.
      }
    }
  });

  describe('checkHealth', () => {
    it('should return true for healthy API', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
      });

      const result = await checkHealth();
      expect(result).toBe(true);
      expect(global.fetch).toHaveBeenCalledWith(
        `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`,
        expect.any(Object)
      );
    });

    it('should return false for unhealthy API', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 500,
      });

      const result = await checkHealth();
      expect(result).toBe(false);
    });

    it('should handle network errors', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      const result = await checkHealth();
      expect(result).toBe(false);
    });
  });

  describe('fetchCurrentGas', () => {
    it('should fetch current gas price successfully', async () => {
      const mockData = {
        timestamp: '2024-01-01T00:00:00Z',
        current_gas: 0.002,
        base_fee: 0.002,
        priority_fee: 0.0001,
        block_number: 12345,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      });

      const result = await fetchCurrentGas();
      expect(result).toEqual(mockData);
    });

    it('should throw error for failed requests', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      });

      await expect(fetchCurrentGas()).rejects.toThrow();
    });
  });

  describe('fetchPredictions', () => {
    it('should fetch predictions successfully', async () => {
      const mockData = {
        current: {
          timestamp: '2024-01-01T00:00:00Z',
          current_gas: 0.002,
        },
        predictions: {
          '1h': [{ predictedGwei: 0.0021 }],
          '4h': [{ predictedGwei: 0.0022 }],
          '24h': [{ predictedGwei: 0.0023 }],
        },
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      });

      const result = await fetchPredictions();
      expect(result).toEqual(mockData);
    });
  });
});
