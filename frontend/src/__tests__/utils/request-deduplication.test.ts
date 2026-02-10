/**
 * Request Deduplication Tests
 *
 * Tests for the request deduplication utility that prevents
 * multiple identical requests from being made simultaneously.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ============================================================================
// Test Implementation of RequestDeduplicator
// ============================================================================

interface PendingRequest<T> {
  promise: Promise<T>;
  timestamp: number;
}

class RequestDeduplicator {
  private pendingRequests: Map<string, PendingRequest<unknown>> = new Map();
  private readonly CACHE_DURATION = 1000; // 1 second

  async deduplicate<T>(key: string, requestFn: () => Promise<T>): Promise<T> {
    const existing = this.pendingRequests.get(key);

    // If request is pending and recent, return existing promise
    if (existing && Date.now() - existing.timestamp < this.CACHE_DURATION) {
      return existing.promise as Promise<T>;
    }

    // Create new request
    const promise = requestFn().finally(() => {
      // Clean up after request completes
      setTimeout(() => {
        this.pendingRequests.delete(key);
      }, this.CACHE_DURATION);
    });

    this.pendingRequests.set(key, {
      promise,
      timestamp: Date.now(),
    });

    return promise;
  }

  clear(): void {
    this.pendingRequests.clear();
  }

  cleanup(): void {
    const now = Date.now();
    for (const [key, request] of this.pendingRequests.entries()) {
      if (now - request.timestamp > this.CACHE_DURATION * 10) {
        this.pendingRequests.delete(key);
      }
    }
  }

  // Test helper: get pending request count
  getPendingCount(): number {
    return this.pendingRequests.size;
  }

  // Test helper: check if key is pending
  hasPending(key: string): boolean {
    return this.pendingRequests.has(key);
  }
}

// ============================================================================
// Tests
// ============================================================================

describe('Request Deduplication Tests', () => {
  let deduplicator: RequestDeduplicator;

  beforeEach(() => {
    deduplicator = new RequestDeduplicator();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    deduplicator.clear();
  });

  describe('Basic Deduplication', () => {
    it('should execute a single request normally', async () => {
      const mockFn = vi.fn().mockResolvedValue('result');

      const result = await deduplicator.deduplicate('key1', mockFn);

      expect(result).toBe('result');
      expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('should deduplicate concurrent identical requests', async () => {
      let resolvePromise: (value: string) => void;
      const promise = new Promise<string>((resolve) => {
        resolvePromise = resolve;
      });
      const mockFn = vi.fn().mockReturnValue(promise);

      // Make two concurrent requests with the same key
      const request1 = deduplicator.deduplicate('same-key', mockFn);
      const request2 = deduplicator.deduplicate('same-key', mockFn);

      // Only one request should be made
      expect(mockFn).toHaveBeenCalledTimes(1);

      // Resolve the promise
      resolvePromise!('shared-result');

      // Both should receive the same result
      const [result1, result2] = await Promise.all([request1, request2]);
      expect(result1).toBe('shared-result');
      expect(result2).toBe('shared-result');
    });

    it('should allow different keys to make separate requests', async () => {
      const mockFn1 = vi.fn().mockResolvedValue('result1');
      const mockFn2 = vi.fn().mockResolvedValue('result2');

      const [result1, result2] = await Promise.all([
        deduplicator.deduplicate('key1', mockFn1),
        deduplicator.deduplicate('key2', mockFn2),
      ]);

      expect(result1).toBe('result1');
      expect(result2).toBe('result2');
      expect(mockFn1).toHaveBeenCalledTimes(1);
      expect(mockFn2).toHaveBeenCalledTimes(1);
    });

    it('should handle many concurrent requests with the same key', async () => {
      let resolvePromise: (value: number) => void;
      const promise = new Promise<number>((resolve) => {
        resolvePromise = resolve;
      });
      const mockFn = vi.fn().mockReturnValue(promise);

      // Make 10 concurrent requests
      const requests = Array.from({ length: 10 }, () =>
        deduplicator.deduplicate('batch-key', mockFn)
      );

      // Only one actual request should be made
      expect(mockFn).toHaveBeenCalledTimes(1);

      resolvePromise!(42);

      // All should receive the same result
      const results = await Promise.all(requests);
      expect(results).toEqual(Array(10).fill(42));
    });
  });

  describe('Cache Duration', () => {
    it('should reuse cached result within cache duration', async () => {
      const mockFn = vi.fn().mockResolvedValue('cached');

      // First request
      await deduplicator.deduplicate('cache-key', mockFn);

      // Second request immediately after (should use cache)
      const mockFn2 = vi.fn().mockResolvedValue('new');
      const result = await deduplicator.deduplicate('cache-key', mockFn2);

      expect(result).toBe('cached');
      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(mockFn2).not.toHaveBeenCalled();
    });

    it('should make new request after cache expires', async () => {
      const mockFn1 = vi.fn().mockResolvedValue('first');
      const mockFn2 = vi.fn().mockResolvedValue('second');

      // First request
      await deduplicator.deduplicate('expiring-key', mockFn1);

      // Advance time past cache duration (1000ms)
      vi.advanceTimersByTime(1100);

      // Second request (cache expired, should make new request)
      const result = await deduplicator.deduplicate('expiring-key', mockFn2);

      expect(result).toBe('second');
      expect(mockFn1).toHaveBeenCalledTimes(1);
      expect(mockFn2).toHaveBeenCalledTimes(1);
    });

    it('should cleanup entry after cache duration', async () => {
      const mockFn = vi.fn().mockResolvedValue('result');

      await deduplicator.deduplicate('cleanup-key', mockFn);

      // Entry should be pending
      expect(deduplicator.hasPending('cleanup-key')).toBe(true);

      // Advance time past cache duration
      vi.advanceTimersByTime(1100);

      // Entry should be cleaned up
      expect(deduplicator.hasPending('cleanup-key')).toBe(false);
    });
  });

  describe('Error Handling', () => {
    it('should propagate errors to all waiting requests', async () => {
      let rejectPromise: (error: Error) => void;
      const promise = new Promise<string>((_, reject) => {
        rejectPromise = reject;
      });
      const mockFn = vi.fn().mockReturnValue(promise);

      // Make concurrent requests
      const request1 = deduplicator.deduplicate('error-key', mockFn);
      const request2 = deduplicator.deduplicate('error-key', mockFn);

      // Reject the promise
      rejectPromise!(new Error('Request failed'));

      // Both should reject with the same error
      await expect(request1).rejects.toThrow('Request failed');
      await expect(request2).rejects.toThrow('Request failed');
    });

    it('should cleanup after error', async () => {
      const mockFn = vi.fn().mockRejectedValue(new Error('Failed'));

      try {
        await deduplicator.deduplicate('error-cleanup-key', mockFn);
      } catch {
        // Expected
      }

      // Advance time for cleanup
      vi.advanceTimersByTime(1100);

      // Entry should be cleaned up
      expect(deduplicator.hasPending('error-cleanup-key')).toBe(false);
    });

    it('should allow retry after error', async () => {
      const mockFn1 = vi.fn().mockRejectedValue(new Error('First failed'));
      const mockFn2 = vi.fn().mockResolvedValue('success');

      // First request fails
      try {
        await deduplicator.deduplicate('retry-key', mockFn1);
      } catch {
        // Expected
      }

      // Wait for cleanup
      vi.advanceTimersByTime(1100);

      // Retry should work
      const result = await deduplicator.deduplicate('retry-key', mockFn2);
      expect(result).toBe('success');
    });
  });

  describe('Clear Method', () => {
    it('should clear all pending requests', async () => {
      const neverResolves = new Promise<string>(() => {});
      const mockFn = vi.fn().mockReturnValue(neverResolves);

      // Start multiple pending requests
      deduplicator.deduplicate('key1', mockFn);
      deduplicator.deduplicate('key2', mockFn);
      deduplicator.deduplicate('key3', mockFn);

      expect(deduplicator.getPendingCount()).toBe(3);

      // Clear all
      deduplicator.clear();

      expect(deduplicator.getPendingCount()).toBe(0);
    });

    it('should allow new requests after clear', async () => {
      const mockFn1 = vi.fn().mockResolvedValue('first');
      const mockFn2 = vi.fn().mockResolvedValue('second');

      await deduplicator.deduplicate('clear-test', mockFn1);

      deduplicator.clear();

      const result = await deduplicator.deduplicate('clear-test', mockFn2);
      expect(result).toBe('second');
      expect(mockFn2).toHaveBeenCalledTimes(1);
    });
  });

  describe('Cleanup Method', () => {
    it('should remove stale requests (older than 10x cache duration)', async () => {
      const neverResolves = new Promise<string>(() => {});
      const mockFn = vi.fn().mockReturnValue(neverResolves);

      // Start a request
      deduplicator.deduplicate('stale-key', mockFn);

      expect(deduplicator.getPendingCount()).toBe(1);

      // Advance time past 10x cache duration (10 seconds)
      vi.advanceTimersByTime(10100);

      // Run cleanup
      deduplicator.cleanup();

      expect(deduplicator.getPendingCount()).toBe(0);
    });

    it('should keep recent requests during cleanup', async () => {
      const neverResolves = new Promise<string>(() => {});
      const mockFn = vi.fn().mockReturnValue(neverResolves);

      // Start a request
      deduplicator.deduplicate('recent-key', mockFn);

      // Advance time but not past stale threshold
      vi.advanceTimersByTime(5000);

      // Run cleanup
      deduplicator.cleanup();

      // Should still be pending
      expect(deduplicator.hasPending('recent-key')).toBe(true);
    });

    it('should handle mixed stale and recent requests', async () => {
      const neverResolves = new Promise<string>(() => {});
      const mockFn = vi.fn().mockReturnValue(neverResolves);

      // Start first request
      deduplicator.deduplicate('old-key', mockFn);

      // Advance time
      vi.advanceTimersByTime(8000);

      // Start second request
      deduplicator.deduplicate('new-key', mockFn);

      // Advance more time (old is now stale)
      vi.advanceTimersByTime(3000);

      // Run cleanup
      deduplicator.cleanup();

      // Old should be gone, new should remain
      expect(deduplicator.hasPending('old-key')).toBe(false);
      expect(deduplicator.hasPending('new-key')).toBe(true);
    });
  });

  describe('Type Safety', () => {
    it('should preserve return types', async () => {
      interface User {
        id: number;
        name: string;
      }

      const mockFn = vi.fn().mockResolvedValue({ id: 1, name: 'Test' });

      const result: User = await deduplicator.deduplicate<User>('user-key', mockFn);

      expect(result.id).toBe(1);
      expect(result.name).toBe('Test');
    });

    it('should work with different return types for different keys', async () => {
      const stringFn = vi.fn().mockResolvedValue('string result');
      const numberFn = vi.fn().mockResolvedValue(42);
      const objectFn = vi.fn().mockResolvedValue({ key: 'value' });

      const [str, num, obj] = await Promise.all([
        deduplicator.deduplicate<string>('string-key', stringFn),
        deduplicator.deduplicate<number>('number-key', numberFn),
        deduplicator.deduplicate<{ key: string }>('object-key', objectFn),
      ]);

      expect(typeof str).toBe('string');
      expect(typeof num).toBe('number');
      expect(typeof obj).toBe('object');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty string key', async () => {
      const mockFn = vi.fn().mockResolvedValue('empty-key-result');

      const result = await deduplicator.deduplicate('', mockFn);

      expect(result).toBe('empty-key-result');
    });

    it('should handle very long keys', async () => {
      const longKey = 'a'.repeat(10000);
      const mockFn = vi.fn().mockResolvedValue('long-key-result');

      const result = await deduplicator.deduplicate(longKey, mockFn);

      expect(result).toBe('long-key-result');
    });

    it('should handle special characters in keys', async () => {
      const specialKeys = ['key/with/slashes', 'key?with=query', 'key#with#hash', 'key with spaces'];

      const results = await Promise.all(
        specialKeys.map((key) =>
          deduplicator.deduplicate(key, () => Promise.resolve(key))
        )
      );

      expect(results).toEqual(specialKeys);
    });

    it('should handle null result', async () => {
      const mockFn = vi.fn().mockResolvedValue(null);

      const result = await deduplicator.deduplicate('null-key', mockFn);

      expect(result).toBeNull();
    });

    it('should handle undefined result', async () => {
      const mockFn = vi.fn().mockResolvedValue(undefined);

      const result = await deduplicator.deduplicate('undefined-key', mockFn);

      expect(result).toBeUndefined();
    });

    it('should handle array results', async () => {
      const mockFn = vi.fn().mockResolvedValue([1, 2, 3]);

      const result = await deduplicator.deduplicate('array-key', mockFn);

      expect(result).toEqual([1, 2, 3]);
    });
  });

  describe('Concurrency Patterns', () => {
    it('should handle rapid sequential calls to same key', async () => {
      let callCount = 0;
      const mockFn = vi.fn().mockImplementation(() => {
        callCount++;
        return Promise.resolve(callCount);
      });

      // Rapid sequential calls
      const result1 = await deduplicator.deduplicate('rapid-key', mockFn);
      const result2 = await deduplicator.deduplicate('rapid-key', mockFn);
      const result3 = await deduplicator.deduplicate('rapid-key', mockFn);

      // All should return the first result due to caching
      expect(result1).toBe(1);
      expect(result2).toBe(1);
      expect(result3).toBe(1);
      expect(mockFn).toHaveBeenCalledTimes(1);
    });

    it('should handle interleaved requests to different keys', async () => {
      const delays = new Map<string, (value: string) => void>();
      const createRequest = (key: string) => {
        return new Promise<string>((resolve) => {
          delays.set(key, resolve);
        });
      };

      const mockFnA = vi.fn().mockReturnValue(createRequest('A'));
      const mockFnB = vi.fn().mockReturnValue(createRequest('B'));
      const mockFnC = vi.fn().mockReturnValue(createRequest('C'));

      // Start requests in order A, B, C
      const requestA = deduplicator.deduplicate('keyA', mockFnA);
      const requestB = deduplicator.deduplicate('keyB', mockFnB);
      const requestC = deduplicator.deduplicate('keyC', mockFnC);

      // Resolve in different order C, A, B
      delays.get('C')!('resultC');
      delays.get('A')!('resultA');
      delays.get('B')!('resultB');

      const [resultA, resultB, resultC] = await Promise.all([requestA, requestB, requestC]);

      expect(resultA).toBe('resultA');
      expect(resultB).toBe('resultB');
      expect(resultC).toBe('resultC');
    });

    it('should handle request that resolves while another starts', async () => {
      let resolveFirst: (value: string) => void;
      const firstPromise = new Promise<string>((resolve) => {
        resolveFirst = resolve;
      });
      const mockFn1 = vi.fn().mockReturnValue(firstPromise);
      const mockFn2 = vi.fn().mockResolvedValue('second');

      // Start first request
      const request1 = deduplicator.deduplicate('overlap-key', mockFn1);

      // Resolve first
      resolveFirst!('first');
      await request1;

      // Start second request immediately (within cache duration)
      const request2 = deduplicator.deduplicate('overlap-key', mockFn2);

      // Should still return cached first result
      const result2 = await request2;
      expect(result2).toBe('first');
      expect(mockFn2).not.toHaveBeenCalled();
    });
  });

  describe('Real-World Scenarios', () => {
    it('should deduplicate API fetch calls', async () => {
      const fetchUser = vi.fn().mockResolvedValue({
        id: 1,
        name: 'John Doe',
        email: 'john@example.com',
      });

      // Simulate multiple components requesting the same user
      const [user1, user2, user3] = await Promise.all([
        deduplicator.deduplicate('user:1', fetchUser),
        deduplicator.deduplicate('user:1', fetchUser),
        deduplicator.deduplicate('user:1', fetchUser),
      ]);

      // All should get the same user with only one API call
      expect(user1).toEqual(user2);
      expect(user2).toEqual(user3);
      expect(fetchUser).toHaveBeenCalledTimes(1);
    });

    it('should handle paginated requests with different keys', async () => {
      const fetchPage = vi.fn().mockImplementation((page: number) =>
        Promise.resolve({ page, items: [`item${page}a`, `item${page}b`] })
      );

      // Fetch different pages
      const [page1, page2, page3] = await Promise.all([
        deduplicator.deduplicate('list:page:1', () => fetchPage(1)),
        deduplicator.deduplicate('list:page:2', () => fetchPage(2)),
        deduplicator.deduplicate('list:page:3', () => fetchPage(3)),
      ]);

      expect(page1.page).toBe(1);
      expect(page2.page).toBe(2);
      expect(page3.page).toBe(3);
      expect(fetchPage).toHaveBeenCalledTimes(3);
    });

    it('should deduplicate search requests', async () => {
      const search = vi.fn().mockResolvedValue([{ id: 1, title: 'Result' }]);

      // User types quickly, triggering multiple search calls with same query
      const query = 'search term';
      const results = await Promise.all([
        deduplicator.deduplicate(`search:${query}`, search),
        deduplicator.deduplicate(`search:${query}`, search),
        deduplicator.deduplicate(`search:${query}`, search),
      ]);

      expect(results[0]).toEqual(results[1]);
      expect(results[1]).toEqual(results[2]);
      expect(search).toHaveBeenCalledTimes(1);
    });

    it('should allow different search queries', async () => {
      const search = vi.fn().mockImplementation((query: string) =>
        Promise.resolve([{ query, results: [] }])
      );

      const [result1, result2] = await Promise.all([
        deduplicator.deduplicate('search:foo', () => search('foo')),
        deduplicator.deduplicate('search:bar', () => search('bar')),
      ]);

      expect(result1[0].query).toBe('foo');
      expect(result2[0].query).toBe('bar');
      expect(search).toHaveBeenCalledTimes(2);
    });
  });
});
