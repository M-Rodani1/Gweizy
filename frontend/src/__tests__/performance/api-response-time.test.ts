/**
 * API Response Time Tests
 *
 * Tests to ensure API calls complete within acceptable time limits
 * for good user experience and performance monitoring.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ============================================================================
// Performance Thresholds (in milliseconds)
// ============================================================================

const PERFORMANCE_THRESHOLDS = {
  // API call thresholds
  API_FAST: 100,       // Fast API calls (cached/simple)
  API_NORMAL: 500,     // Normal API calls
  API_SLOW: 2000,      // Slow API calls (complex queries)
  API_TIMEOUT: 10000,  // Maximum timeout

  // User experience thresholds
  UX_INSTANT: 100,     // Feels instant
  UX_FAST: 300,        // Fast, no spinner needed
  UX_LOADING: 1000,    // Show loading indicator
  UX_FRUSTRATED: 3000, // Users get frustrated

  // Batch operation thresholds
  BATCH_PER_ITEM: 50,  // Time per item in batch
  BATCH_OVERHEAD: 200, // Batch setup overhead
};

// ============================================================================
// Performance Measurement Utilities
// ============================================================================

interface PerformanceResult {
  duration: number;
  timestamp: number;
  success: boolean;
  error?: string;
}

interface PerformanceStats {
  min: number;
  max: number;
  avg: number;
  median: number;
  p90: number;
  p95: number;
  p99: number;
  count: number;
}

/**
 * Measure the duration of an async operation
 */
async function measureDuration<T>(operation: () => Promise<T>): Promise<{ result: T; duration: number }> {
  const start = performance.now();
  const result = await operation();
  const duration = performance.now() - start;
  return { result, duration };
}

/**
 * Measure multiple runs and calculate statistics
 */
async function measureMultiple<T>(
  operation: () => Promise<T>,
  runs: number
): Promise<{ results: T[]; stats: PerformanceStats }> {
  const durations: number[] = [];
  const results: T[] = [];

  for (let i = 0; i < runs; i++) {
    const { result, duration } = await measureDuration(operation);
    results.push(result);
    durations.push(duration);
  }

  const sorted = [...durations].sort((a, b) => a - b);
  const sum = durations.reduce((a, b) => a + b, 0);

  const stats: PerformanceStats = {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    avg: sum / runs,
    median: sorted[Math.floor(runs / 2)],
    p90: sorted[Math.floor(runs * 0.9)],
    p95: sorted[Math.floor(runs * 0.95)],
    p99: sorted[Math.floor(runs * 0.99)],
    count: runs,
  };

  return { results, stats };
}

/**
 * Create a mock API with configurable delay
 */
function createMockApi(baseDelay: number, variance = 0) {
  return async <T>(response: T): Promise<T> => {
    const delay = baseDelay + (Math.random() * variance * 2 - variance);
    await new Promise((r) => setTimeout(r, delay));
    return response;
  };
}

/**
 * Assert that duration is within threshold
 */
function assertWithinThreshold(duration: number, threshold: number, tolerance = 0.1): void {
  const maxAllowed = threshold * (1 + tolerance);
  expect(duration).toBeLessThanOrEqual(maxAllowed);
}

// ============================================================================
// Tests
// ============================================================================

describe('API Response Time Tests', () => {
  describe('Performance Measurement Utilities', () => {
    it('should measure operation duration accurately', async () => {
      const targetDelay = 50;
      const { duration } = await measureDuration(async () => {
        await new Promise((r) => setTimeout(r, targetDelay));
        return 'done';
      });

      // Allow 20% tolerance for timer precision
      expect(duration).toBeGreaterThanOrEqual(targetDelay * 0.8);
      expect(duration).toBeLessThan(targetDelay * 1.5);
    });

    it('should calculate statistics correctly', async () => {
      let callCount = 0;
      const { stats } = await measureMultiple(async () => {
        callCount++;
        await new Promise((r) => setTimeout(r, 10));
        return callCount;
      }, 10);

      expect(stats.count).toBe(10);
      expect(stats.min).toBeGreaterThan(0);
      expect(stats.max).toBeGreaterThanOrEqual(stats.min);
      expect(stats.avg).toBeGreaterThan(0);
      expect(stats.median).toBeGreaterThan(0);
    });

    it('should handle zero-duration operations', async () => {
      const { duration } = await measureDuration(async () => {
        return 'instant';
      });

      expect(duration).toBeLessThan(10); // Should be nearly instant
    });
  });

  describe('API Response Time Thresholds', () => {
    it('should define reasonable thresholds', () => {
      expect(PERFORMANCE_THRESHOLDS.API_FAST).toBeLessThan(PERFORMANCE_THRESHOLDS.API_NORMAL);
      expect(PERFORMANCE_THRESHOLDS.API_NORMAL).toBeLessThan(PERFORMANCE_THRESHOLDS.API_SLOW);
      expect(PERFORMANCE_THRESHOLDS.API_SLOW).toBeLessThan(PERFORMANCE_THRESHOLDS.API_TIMEOUT);
    });

    it('should have UX thresholds aligned with best practices', () => {
      // Instant response (< 100ms) - feels immediate
      expect(PERFORMANCE_THRESHOLDS.UX_INSTANT).toBe(100);

      // Fast response (< 300ms) - no spinner needed
      expect(PERFORMANCE_THRESHOLDS.UX_FAST).toBe(300);

      // Loading state (< 1000ms) - show loading indicator
      expect(PERFORMANCE_THRESHOLDS.UX_LOADING).toBe(1000);

      // Frustration threshold (< 3000ms) - users start abandoning
      expect(PERFORMANCE_THRESHOLDS.UX_FRUSTRATED).toBe(3000);
    });
  });

  describe('Mock API Response Times', () => {
    it('should complete fast API calls within threshold', async () => {
      const fastApi = createMockApi(50);
      const { duration } = await measureDuration(() => fastApi({ data: 'fast' }));

      assertWithinThreshold(duration, PERFORMANCE_THRESHOLDS.API_FAST);
    });

    it('should complete normal API calls within threshold', async () => {
      const normalApi = createMockApi(200);
      const { duration } = await measureDuration(() => normalApi({ data: 'normal' }));

      assertWithinThreshold(duration, PERFORMANCE_THRESHOLDS.API_NORMAL);
    });

    it('should handle variable response times', async () => {
      const variableApi = createMockApi(100, 50);
      const { stats } = await measureMultiple(() => variableApi({ data: 'variable' }), 10);

      // All responses should be within bounds
      expect(stats.min).toBeGreaterThanOrEqual(50);
      expect(stats.max).toBeLessThanOrEqual(200);
    });
  });

  describe('Parallel Request Performance', () => {
    it('should handle parallel requests efficiently', async () => {
      const api = createMockApi(50);
      const requests = Array.from({ length: 5 }, (_, i) => api({ id: i }));

      const { duration } = await measureDuration(() => Promise.all(requests));

      // Parallel requests should complete in roughly the same time as single request
      // (with some overhead for Promise.all)
      expect(duration).toBeLessThan(100);
    });

    it('should not increase linearly with parallel requests', async () => {
      const api = createMockApi(30);

      const { duration: single } = await measureDuration(() => api({ id: 1 }));
      const { duration: parallel } = await measureDuration(() =>
        Promise.all(Array.from({ length: 10 }, (_, i) => api({ id: i })))
      );

      // Parallel should not be 10x slower
      expect(parallel).toBeLessThan(single * 3);
    });
  });

  describe('Sequential Request Performance', () => {
    it('should measure sequential request chain', async () => {
      const api = createMockApi(20);
      const chainLength = 5;

      const { duration } = await measureDuration(async () => {
        let result = { id: 0 };
        for (let i = 1; i <= chainLength; i++) {
          result = await api({ id: i });
        }
        return result;
      });

      // Sequential requests should take approximately n * delay
      const expectedMin = chainLength * 20 * 0.8;
      const expectedMax = chainLength * 20 * 1.5;

      expect(duration).toBeGreaterThanOrEqual(expectedMin);
      expect(duration).toBeLessThanOrEqual(expectedMax);
    });
  });

  describe('Timeout Handling', () => {
    it('should detect operations exceeding timeout', async () => {
      const timeout = 100;
      const slowOperation = async () => {
        await new Promise((r) => setTimeout(r, 200));
        return 'done';
      };

      const { duration } = await measureDuration(slowOperation);
      const exceededTimeout = duration > timeout;

      expect(exceededTimeout).toBe(true);
    });

    it('should cancel operations on timeout', async () => {
      const timeout = 50;
      let completed = false;

      const operation = async (signal: AbortSignal) => {
        return new Promise((resolve, reject) => {
          const timeoutId = setTimeout(() => {
            completed = true;
            resolve('done');
          }, 200);

          signal.addEventListener('abort', () => {
            clearTimeout(timeoutId);
            reject(new Error('Aborted'));
          });
        });
      };

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      try {
        await operation(controller.signal);
      } catch {
        // Expected
      } finally {
        clearTimeout(timeoutId);
      }

      expect(completed).toBe(false);
    });
  });

  describe('Response Time Categories', () => {
    type ResponseCategory = 'instant' | 'fast' | 'loading' | 'slow' | 'timeout';

    function categorizeResponseTime(duration: number): ResponseCategory {
      if (duration <= PERFORMANCE_THRESHOLDS.UX_INSTANT) return 'instant';
      if (duration <= PERFORMANCE_THRESHOLDS.UX_FAST) return 'fast';
      if (duration <= PERFORMANCE_THRESHOLDS.UX_LOADING) return 'loading';
      if (duration <= PERFORMANCE_THRESHOLDS.UX_FRUSTRATED) return 'slow';
      return 'timeout';
    }

    it('should categorize instant responses', () => {
      expect(categorizeResponseTime(50)).toBe('instant');
      expect(categorizeResponseTime(99)).toBe('instant');
    });

    it('should categorize fast responses', () => {
      expect(categorizeResponseTime(101)).toBe('fast');
      expect(categorizeResponseTime(299)).toBe('fast');
    });

    it('should categorize loading responses', () => {
      expect(categorizeResponseTime(301)).toBe('loading');
      expect(categorizeResponseTime(999)).toBe('loading');
    });

    it('should categorize slow responses', () => {
      expect(categorizeResponseTime(1001)).toBe('slow');
      expect(categorizeResponseTime(2999)).toBe('slow');
    });

    it('should categorize timeout responses', () => {
      expect(categorizeResponseTime(3001)).toBe('timeout');
      expect(categorizeResponseTime(10000)).toBe('timeout');
    });
  });

  describe('Performance Budget', () => {
    interface PerformanceBudget {
      totalBudget: number;
      operations: Array<{ name: string; budget: number }>;
    }

    function checkBudget(
      budget: PerformanceBudget,
      measurements: Map<string, number>
    ): { passed: boolean; details: Array<{ name: string; budget: number; actual: number; passed: boolean }> } {
      const details = budget.operations.map((op) => {
        const actual = measurements.get(op.name) || 0;
        return {
          name: op.name,
          budget: op.budget,
          actual,
          passed: actual <= op.budget,
        };
      });

      const totalActual = Array.from(measurements.values()).reduce((a, b) => a + b, 0);
      const passed = details.every((d) => d.passed) && totalActual <= budget.totalBudget;

      return { passed, details };
    }

    it('should track performance budget', () => {
      const budget: PerformanceBudget = {
        totalBudget: 1000,
        operations: [
          { name: 'fetch-data', budget: 500 },
          { name: 'process-data', budget: 200 },
          { name: 'render', budget: 300 },
        ],
      };

      const measurements = new Map([
        ['fetch-data', 400],
        ['process-data', 150],
        ['render', 250],
      ]);

      const result = checkBudget(budget, measurements);

      expect(result.passed).toBe(true);
      expect(result.details).toHaveLength(3);
      expect(result.details.every((d) => d.passed)).toBe(true);
    });

    it('should detect budget violations', () => {
      const budget: PerformanceBudget = {
        totalBudget: 500,
        operations: [
          { name: 'slow-fetch', budget: 200 },
          { name: 'slow-process', budget: 200 },
        ],
      };

      const measurements = new Map([
        ['slow-fetch', 300], // Over budget
        ['slow-process', 250], // Over budget
      ]);

      const result = checkBudget(budget, measurements);

      expect(result.passed).toBe(false);
    });
  });

  describe('Batch Operation Performance', () => {
    it('should calculate expected batch duration', () => {
      const itemCount = 10;
      const expectedDuration =
        PERFORMANCE_THRESHOLDS.BATCH_OVERHEAD +
        itemCount * PERFORMANCE_THRESHOLDS.BATCH_PER_ITEM;

      expect(expectedDuration).toBe(700); // 200 + 10*50
    });

    it('should process batch within calculated time', async () => {
      const items = Array.from({ length: 5 }, (_, i) => i);
      const api = createMockApi(30);

      const expectedMax =
        PERFORMANCE_THRESHOLDS.BATCH_OVERHEAD +
        items.length * PERFORMANCE_THRESHOLDS.BATCH_PER_ITEM;

      const { duration } = await measureDuration(async () => {
        // Process in parallel
        return Promise.all(items.map((item) => api({ id: item })));
      });

      expect(duration).toBeLessThan(expectedMax);
    });
  });

  describe('Performance Regression Detection', () => {
    interface PerformanceBaseline {
      operation: string;
      p50: number;
      p95: number;
      p99: number;
    }

    function detectRegression(
      baseline: PerformanceBaseline,
      current: PerformanceStats,
      threshold = 0.2
    ): { hasRegression: boolean; details: string[] } {
      const details: string[] = [];
      let hasRegression = false;

      if (current.median > baseline.p50 * (1 + threshold)) {
        details.push(`P50 regression: baseline ${baseline.p50}ms, current ${current.median.toFixed(1)}ms`);
        hasRegression = true;
      }

      if (current.p95 > baseline.p95 * (1 + threshold)) {
        details.push(`P95 regression: baseline ${baseline.p95}ms, current ${current.p95.toFixed(1)}ms`);
        hasRegression = true;
      }

      if (current.p99 > baseline.p99 * (1 + threshold)) {
        details.push(`P99 regression: baseline ${baseline.p99}ms, current ${current.p99.toFixed(1)}ms`);
        hasRegression = true;
      }

      return { hasRegression, details };
    }

    it('should detect performance regression', () => {
      const baseline: PerformanceBaseline = {
        operation: 'api-call',
        p50: 100,
        p95: 200,
        p99: 500,
      };

      const regressedStats: PerformanceStats = {
        min: 50,
        max: 1000,
        avg: 150,
        median: 150, // 50% increase
        p90: 250,
        p95: 300, // 50% increase
        p99: 800, // 60% increase
        count: 100,
      };

      const result = detectRegression(baseline, regressedStats);

      expect(result.hasRegression).toBe(true);
      expect(result.details.length).toBeGreaterThan(0);
    });

    it('should pass when within acceptable range', () => {
      const baseline: PerformanceBaseline = {
        operation: 'api-call',
        p50: 100,
        p95: 200,
        p99: 500,
      };

      const goodStats: PerformanceStats = {
        min: 50,
        max: 550,
        avg: 100,
        median: 110, // 10% increase (within threshold)
        p90: 180,
        p95: 220, // 10% increase
        p99: 540, // 8% increase
        count: 100,
      };

      const result = detectRegression(baseline, goodStats);

      expect(result.hasRegression).toBe(false);
    });
  });
});
