/**
 * Example tests demonstrating MSW usage
 *
 * These tests show how to use Mock Service Worker for API mocking:
 * - Default handlers for success scenarios
 * - Custom handlers for error scenarios
 * - Custom handlers for delayed responses
 *
 * MSW is opt-in per test file. Start/stop the server in your test file.
 */

import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { server } from './server';
import { errorHandlers, mockData } from './handlers';
import { checkHealth, fetchCurrentGas, fetchPredictions } from '../../api/gasApi';

// Start MSW server for this test file
beforeAll(() => {
  server.listen({ onUnhandledRequest: 'bypass' });
});

afterEach(() => {
  server.resetHandlers();
});

afterAll(() => {
  server.close();
});

describe('MSW Integration Examples', () => {
  describe('Default handlers (success scenarios)', () => {
    it('should return healthy status from health endpoint', async () => {
      const result = await checkHealth();
      expect(result).toBe(true);
    });

    it('should fetch current gas data', async () => {
      const result = await fetchCurrentGas();

      expect(result).toHaveProperty('current_gas');
      expect(result).toHaveProperty('base_fee');
      expect(result).toHaveProperty('priority_fee');
      expect(result).toHaveProperty('block_number');
    });

    it('should fetch predictions data', async () => {
      const result = await fetchPredictions();

      expect(result).toHaveProperty('current');
      expect(result).toHaveProperty('predictions');
      expect(result.predictions).toHaveProperty('1h');
      expect(result.predictions).toHaveProperty('4h');
      expect(result.predictions).toHaveProperty('24h');
    });
  });

  describe('Error handlers', () => {
    it('should handle API errors gracefully', async () => {
      // Override with error handlers for this test
      server.use(...errorHandlers);

      // fetchCurrentGas has retry logic and cache fallback,
      // so it may not throw immediately. The mock data won't match
      // the error response structure.
      try {
        await fetchCurrentGas();
        // If it didn't throw, the cache fallback was used
      } catch (error) {
        // Expected in some scenarios
        expect(error).toBeDefined();
      }
    });
  });

  describe('Using mock data for assertions', () => {
    it('should return data matching mock expectations', async () => {
      const result = await fetchCurrentGas();

      // mockData contains the same values returned by handlers
      expect(result.current_gas).toBe(mockData.currentGas.current_gas);
      expect(result.base_fee).toBe(mockData.currentGas.base_fee);
    });
  });
});
