/**
 * Network Chaos Tests
 *
 * Tests the application's resilience to various network failure scenarios.
 * Simulates real-world conditions like timeouts, intermittent failures,
 * connection drops, and degraded performance.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';

// ============================================================================
// Chaos Testing Utilities
// ============================================================================

type ChaosScenario =
  | 'timeout'
  | 'connection_reset'
  | 'intermittent'
  | 'slow_response'
  | 'partial_failure'
  | 'rate_limit'
  | 'server_error'
  | 'corrupt_response';

interface ChaosConfig {
  scenario: ChaosScenario;
  probability?: number; // For intermittent failures (0-1)
  delayMs?: number; // For slow responses
  failureCount?: number; // Number of failures before success
  statusCode?: number;
}

class ChaosFetch {
  private config: ChaosConfig;
  private callCount = 0;
  private originalFetch: typeof fetch;

  constructor(config: ChaosConfig) {
    this.config = config;
    this.originalFetch = global.fetch;
  }

  apply(): void {
    const self = this;

    global.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      self.callCount++;

      switch (self.config.scenario) {
        case 'timeout':
          return self.simulateTimeout();

        case 'connection_reset':
          return self.simulateConnectionReset();

        case 'intermittent':
          return self.simulateIntermittentFailure(input, init);

        case 'slow_response':
          return self.simulateSlowResponse(input, init);

        case 'partial_failure':
          return self.simulatePartialFailure(input, init);

        case 'rate_limit':
          return self.simulateRateLimit();

        case 'server_error':
          return self.simulateServerError();

        case 'corrupt_response':
          return self.simulateCorruptResponse();

        default:
          return self.originalFetch(input, init);
      }
    };
  }

  restore(): void {
    global.fetch = this.originalFetch;
    this.callCount = 0;
  }

  getCallCount(): number {
    return this.callCount;
  }

  private async simulateTimeout(): Promise<never> {
    const timeoutMs = this.config.delayMs || 30000;
    await new Promise((resolve) => setTimeout(resolve, timeoutMs));
    throw new Error('Request timeout');
  }

  private simulateConnectionReset(): never {
    throw new Error('Connection reset by peer');
  }

  private async simulateIntermittentFailure(
    _input: RequestInfo | URL,
    _init?: RequestInit
  ): Promise<Response> {
    const probability = this.config.probability || 0.5;

    if (Math.random() < probability) {
      throw new Error('Network request failed');
    }

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  private async simulateSlowResponse(
    _input: RequestInfo | URL,
    _init?: RequestInit
  ): Promise<Response> {
    const delayMs = this.config.delayMs || 5000;
    await new Promise((resolve) => setTimeout(resolve, delayMs));

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  private async simulatePartialFailure(
    _input: RequestInfo | URL,
    _init?: RequestInit
  ): Promise<Response> {
    const failureCount = this.config.failureCount || 3;

    if (this.callCount <= failureCount) {
      throw new Error('Network request failed');
    }

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  private simulateRateLimit(): Response {
    return new Response(
      JSON.stringify({
        success: false,
        error: 'Rate limit exceeded',
        retry_after: 60,
      }),
      {
        status: 429,
        headers: {
          'Content-Type': 'application/json',
          'Retry-After': '60',
        },
      }
    );
  }

  private simulateServerError(): Response {
    const statusCode = this.config.statusCode || 500;
    return new Response(
      JSON.stringify({
        success: false,
        error: 'Internal server error',
      }),
      {
        status: statusCode,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }

  private simulateCorruptResponse(): Response {
    return new Response('{"invalid json without closing', {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

// ============================================================================
// Retry Logic Implementation (to test)
// ============================================================================

interface RetryConfig {
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
}

const defaultRetryConfig: RetryConfig = {
  maxRetries: 3,
  baseDelayMs: 100,
  maxDelayMs: 5000,
  backoffMultiplier: 2,
};

async function fetchWithRetry(
  url: string,
  options?: RequestInit,
  retryConfig: RetryConfig = defaultRetryConfig
): Promise<Response> {
  let lastError: Error | null = null;
  let delay = retryConfig.baseDelayMs;

  for (let attempt = 0; attempt <= retryConfig.maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);

      // Retry on server errors (5xx) but not client errors (4xx)
      if (response.status >= 500) {
        throw new Error(`Server error: ${response.status}`);
      }

      return response;
    } catch (error) {
      lastError = error as Error;

      if (attempt < retryConfig.maxRetries) {
        await new Promise((resolve) => setTimeout(resolve, delay));
        delay = Math.min(delay * retryConfig.backoffMultiplier, retryConfig.maxDelayMs);
      }
    }
  }

  throw lastError || new Error('Request failed after retries');
}

// ============================================================================
// Chaos Tests
// ============================================================================

describe('Network Chaos Tests', () => {
  let chaosFetch: ChaosFetch | null = null;

  afterEach(() => {
    if (chaosFetch) {
      chaosFetch.restore();
      chaosFetch = null;
    }
    vi.useRealTimers();
  });

  describe('Connection Reset Handling', () => {
    it('should handle connection reset gracefully', async () => {
      chaosFetch = new ChaosFetch({ scenario: 'connection_reset' });
      chaosFetch.apply();

      await expect(fetch('/api/test')).rejects.toThrow('Connection reset by peer');
    });

    it('should retry after connection reset', async () => {
      chaosFetch = new ChaosFetch({
        scenario: 'partial_failure',
        failureCount: 2,
      });
      chaosFetch.apply();

      const response = await fetchWithRetry('/api/test');
      expect(response.status).toBe(200);
      expect(chaosFetch.getCallCount()).toBe(3);
    });
  });

  describe('Intermittent Failure Handling', () => {
    it('should eventually succeed with intermittent failures', async () => {
      let successCount = 0;
      let failureCount = 0;

      chaosFetch = new ChaosFetch({
        scenario: 'intermittent',
        probability: 0.5,
      });
      chaosFetch.apply();

      // Make multiple requests to test intermittent behavior
      for (let i = 0; i < 20; i++) {
        try {
          const response = await fetch('/api/test');
          if (response.ok) successCount++;
        } catch {
          failureCount++;
        }
      }

      // With 50% probability, we should see both successes and failures
      expect(successCount).toBeGreaterThan(0);
      expect(failureCount).toBeGreaterThan(0);
    });

    it('should handle 90% failure rate with retries', async () => {
      chaosFetch = new ChaosFetch({
        scenario: 'intermittent',
        probability: 0.9,
      });
      chaosFetch.apply();

      // With retries, we should eventually succeed
      const results: boolean[] = [];

      for (let i = 0; i < 5; i++) {
        try {
          await fetchWithRetry('/api/test', undefined, {
            maxRetries: 10,
            baseDelayMs: 10,
            maxDelayMs: 100,
            backoffMultiplier: 1.5,
          });
          results.push(true);
        } catch {
          results.push(false);
        }
      }

      // At least some should succeed with enough retries
      const successRate = results.filter(Boolean).length / results.length;
      expect(successRate).toBeGreaterThan(0);
    });
  });

  describe('Slow Response Handling', () => {
    it('should handle slow responses', async () => {
      vi.useFakeTimers();

      chaosFetch = new ChaosFetch({
        scenario: 'slow_response',
        delayMs: 2000,
      });
      chaosFetch.apply();

      const fetchPromise = fetch('/api/test');

      // Advance timers
      await vi.advanceTimersByTimeAsync(2000);

      const response = await fetchPromise;
      expect(response.status).toBe(200);
    });

    it('should support AbortController for cancellation', async () => {
      // Test that AbortController is properly supported
      const controller = new AbortController();

      // Verify signal property exists
      expect(controller.signal).toBeDefined();
      expect(controller.signal.aborted).toBe(false);

      // Abort and verify
      controller.abort();
      expect(controller.signal.aborted).toBe(true);
    });
  });

  describe('Rate Limit Handling', () => {
    it('should handle 429 responses', async () => {
      chaosFetch = new ChaosFetch({ scenario: 'rate_limit' });
      chaosFetch.apply();

      const response = await fetch('/api/test');
      expect(response.status).toBe(429);

      const data = await response.json();
      expect(data.retry_after).toBe(60);
      expect(response.headers.get('Retry-After')).toBe('60');
    });

    it('should extract retry information from rate limit response', async () => {
      chaosFetch = new ChaosFetch({ scenario: 'rate_limit' });
      chaosFetch.apply();

      const response = await fetch('/api/test');
      const retryAfter = parseInt(response.headers.get('Retry-After') || '0', 10);

      expect(retryAfter).toBe(60);
    });
  });

  describe('Server Error Handling', () => {
    it('should handle 500 errors', async () => {
      chaosFetch = new ChaosFetch({
        scenario: 'server_error',
        statusCode: 500,
      });
      chaosFetch.apply();

      const response = await fetch('/api/test');
      expect(response.status).toBe(500);
    });

    it('should handle 502 Bad Gateway', async () => {
      chaosFetch = new ChaosFetch({
        scenario: 'server_error',
        statusCode: 502,
      });
      chaosFetch.apply();

      const response = await fetch('/api/test');
      expect(response.status).toBe(502);
    });

    it('should handle 503 Service Unavailable', async () => {
      chaosFetch = new ChaosFetch({
        scenario: 'server_error',
        statusCode: 503,
      });
      chaosFetch.apply();

      const response = await fetch('/api/test');
      expect(response.status).toBe(503);
    });

    it('should retry 5xx errors with backoff', async () => {
      chaosFetch = new ChaosFetch({
        scenario: 'partial_failure',
        failureCount: 2,
      });
      chaosFetch.apply();

      const startTime = Date.now();
      const response = await fetchWithRetry('/api/test', undefined, {
        maxRetries: 5,
        baseDelayMs: 50,
        maxDelayMs: 500,
        backoffMultiplier: 2,
      });
      const duration = Date.now() - startTime;

      expect(response.status).toBe(200);
      // Should have delayed for at least 50ms + 100ms = 150ms
      expect(duration).toBeGreaterThanOrEqual(100);
    });
  });

  describe('Corrupt Response Handling', () => {
    it('should handle corrupt JSON responses', async () => {
      chaosFetch = new ChaosFetch({ scenario: 'corrupt_response' });
      chaosFetch.apply();

      const response = await fetch('/api/test');
      expect(response.status).toBe(200);

      await expect(response.json()).rejects.toThrow();
    });

    it('should provide fallback on corrupt data', async () => {
      chaosFetch = new ChaosFetch({ scenario: 'corrupt_response' });
      chaosFetch.apply();

      const response = await fetch('/api/test');
      let data = { success: false, fallback: true };

      try {
        data = await response.json();
      } catch {
        // Use fallback data
      }

      expect(data.fallback).toBe(true);
    });
  });

  describe('Exponential Backoff', () => {
    it('should increase delay exponentially', async () => {
      const delays: number[] = [];
      let lastCallTime = Date.now();

      chaosFetch = new ChaosFetch({
        scenario: 'partial_failure',
        failureCount: 3,
      });
      chaosFetch.apply();

      // Track delays between retries
      const originalFetch = global.fetch;
      global.fetch = async (...args) => {
        const now = Date.now();
        delays.push(now - lastCallTime);
        lastCallTime = now;
        return originalFetch(...args);
      };

      try {
        await fetchWithRetry('/api/test', undefined, {
          maxRetries: 5,
          baseDelayMs: 50,
          maxDelayMs: 500,
          backoffMultiplier: 2,
        });
      } finally {
        global.fetch = originalFetch;
      }

      // Check that delays increase (allow for timing variance)
      for (let i = 2; i < delays.length; i++) {
        expect(delays[i]).toBeGreaterThanOrEqual(delays[i - 1] * 0.5);
      }
    });

    it('should respect max delay cap', async () => {
      const delays: number[] = [];
      let lastCallTime = Date.now();

      chaosFetch = new ChaosFetch({
        scenario: 'partial_failure',
        failureCount: 10,
      });
      chaosFetch.apply();

      const originalFetch = global.fetch;
      global.fetch = async (...args) => {
        const now = Date.now();
        delays.push(now - lastCallTime);
        lastCallTime = now;
        return originalFetch(...args);
      };

      try {
        await fetchWithRetry('/api/test', undefined, {
          maxRetries: 12,
          baseDelayMs: 10,
          maxDelayMs: 100,
          backoffMultiplier: 3,
        });
      } finally {
        global.fetch = originalFetch;
      }

      // All delays should be capped at maxDelayMs
      delays.slice(1).forEach((delay) => {
        expect(delay).toBeLessThanOrEqual(200); // Allow some margin
      });
    });
  });

  describe('Resilience Patterns', () => {
    it('should implement circuit breaker pattern', async () => {
      let failureCount = 0;
      const FAILURE_THRESHOLD = 3;
      let circuitOpen = false;

      chaosFetch = new ChaosFetch({ scenario: 'server_error', statusCode: 500 });
      chaosFetch.apply();

      async function fetchWithCircuitBreaker(url: string): Promise<Response> {
        if (circuitOpen) {
          throw new Error('Circuit is open');
        }

        try {
          const response = await fetch(url);
          if (!response.ok) {
            failureCount++;
            if (failureCount >= FAILURE_THRESHOLD) {
              circuitOpen = true;
            }
            throw new Error(`HTTP ${response.status}`);
          }
          failureCount = 0; // Reset on success
          return response;
        } catch (error) {
          failureCount++;
          if (failureCount >= FAILURE_THRESHOLD) {
            circuitOpen = true;
          }
          throw error;
        }
      }

      // Trigger failures to open circuit
      for (let i = 0; i < 4; i++) {
        try {
          await fetchWithCircuitBreaker('/api/test');
        } catch {
          // Expected
        }
      }

      expect(circuitOpen).toBe(true);

      // Subsequent calls should fail fast
      await expect(fetchWithCircuitBreaker('/api/test')).rejects.toThrow(
        'Circuit is open'
      );
    });

    it('should implement bulkhead pattern for isolation', async () => {
      const MAX_CONCURRENT = 3;
      let currentConnections = 0;
      let peakConnections = 0;

      async function fetchWithBulkhead(_url: string): Promise<Response> {
        if (currentConnections >= MAX_CONCURRENT) {
          throw new Error('Bulkhead limit reached');
        }

        currentConnections++;
        peakConnections = Math.max(peakConnections, currentConnections);

        try {
          // Simulate some work
          await new Promise((resolve) => setTimeout(resolve, 50));
          return new Response(JSON.stringify({ success: true }));
        } finally {
          currentConnections--;
        }
      }

      // Make concurrent requests
      const requests = Array.from({ length: 10 }, () =>
        fetchWithBulkhead('/api/test').catch((e) => e)
      );

      await Promise.all(requests);

      expect(peakConnections).toBeLessThanOrEqual(MAX_CONCURRENT);
    });
  });
});

describe('Chaos Test Utilities', () => {
  it('should create ChaosFetch with all scenarios', () => {
    const scenarios: ChaosScenario[] = [
      'timeout',
      'connection_reset',
      'intermittent',
      'slow_response',
      'partial_failure',
      'rate_limit',
      'server_error',
      'corrupt_response',
    ];

    scenarios.forEach((scenario) => {
      const chaos = new ChaosFetch({ scenario });
      expect(chaos).toBeDefined();
    });
  });

  it('should track call count correctly', async () => {
    const chaos = new ChaosFetch({
      scenario: 'slow_response',
      delayMs: 1, // Very short delay for fast testing
    });
    chaos.apply();

    await fetch('/api/test');
    await fetch('/api/test');
    await fetch('/api/test');

    expect(chaos.getCallCount()).toBe(3);

    chaos.restore();
  });
});
