/**
 * Tests for request batching utilities
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  createBatcher,
  createPredictionBatcher,
  createChainBatcher,
  RateLimitedQueue,
} from '../../utils/requestBatcher';

describe('createBatcher', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('basic batching', () => {
    it('should batch multiple requests', async () => {
      const batchFn = vi.fn().mockResolvedValue(
        new Map([
          [1, 'result1'],
          [2, 'result2'],
          [3, 'result3'],
        ])
      );

      const batcher = createBatcher<number, string>({ batchFn });

      const promise1 = batcher.request(1);
      const promise2 = batcher.request(2);
      const promise3 = batcher.request(3);

      // Advance timers to trigger batch execution
      await vi.advanceTimersByTimeAsync(20);

      const results = await Promise.all([promise1, promise2, promise3]);

      expect(batchFn).toHaveBeenCalledTimes(1);
      expect(batchFn).toHaveBeenCalledWith([1, 2, 3]);
      expect(results).toEqual(['result1', 'result2', 'result3']);
    });

    it('should return correct results for each key', async () => {
      const batchFn = vi.fn().mockResolvedValue(
        new Map([
          ['a', 100],
          ['b', 200],
          ['c', 300],
        ])
      );

      const batcher = createBatcher<string, number>({ batchFn });

      const promiseA = batcher.request('a');
      const promiseB = batcher.request('b');
      const promiseC = batcher.request('c');

      await vi.advanceTimersByTimeAsync(20);

      expect(await promiseA).toBe(100);
      expect(await promiseB).toBe(200);
      expect(await promiseC).toBe(300);
    });

    it('should respect maxWait timing', async () => {
      const batchFn = vi.fn().mockResolvedValue(new Map([[1, 'result']]));

      const batcher = createBatcher<number, string>({
        batchFn,
        maxWait: 50,
      });

      batcher.request(1);

      // Should not execute before maxWait
      await vi.advanceTimersByTimeAsync(40);
      expect(batchFn).not.toHaveBeenCalled();

      // Should execute after maxWait
      await vi.advanceTimersByTimeAsync(15);
      expect(batchFn).toHaveBeenCalledTimes(1);
    });

    it('should force execution at maxBatchSize', async () => {
      const batchFn = vi.fn().mockImplementation((keys: number[]) => {
        return Promise.resolve(new Map(keys.map((k) => [k, `result${k}`])));
      });

      const batcher = createBatcher<number, string>({
        batchFn,
        maxWait: 1000, // Long wait
        maxBatchSize: 3,
      });

      batcher.request(1);
      batcher.request(2);
      expect(batchFn).not.toHaveBeenCalled();

      batcher.request(3); // Should trigger batch

      // Allow promise microtask to run
      await vi.advanceTimersByTimeAsync(0);

      expect(batchFn).toHaveBeenCalledTimes(1);
      expect(batchFn).toHaveBeenCalledWith([1, 2, 3]);
    });
  });

  describe('deduplication', () => {
    it('should deduplicate identical keys', async () => {
      const batchFn = vi.fn().mockResolvedValue(new Map([[1, 'result']]));

      const batcher = createBatcher<number, string>({ batchFn });

      const promise1 = batcher.request(1);
      const promise2 = batcher.request(1);
      const promise3 = batcher.request(1);

      await vi.advanceTimersByTimeAsync(20);

      const results = await Promise.all([promise1, promise2, promise3]);

      expect(batchFn).toHaveBeenCalledTimes(1);
      expect(batchFn).toHaveBeenCalledWith([1]); // Only one key
      expect(results).toEqual(['result', 'result', 'result']);
    });
  });

  describe('error handling', () => {
    it('should reject all requests on batch failure', async () => {
      const error = new Error('Batch failed');
      const batchFn = vi.fn().mockRejectedValue(error);

      const batcher = createBatcher<number, string>({ batchFn });

      const promise1 = batcher.request(1);
      const promise2 = batcher.request(2);

      await vi.advanceTimersByTimeAsync(20);

      await expect(promise1).rejects.toThrow('Batch failed');
      await expect(promise2).rejects.toThrow('Batch failed');
    });

    it('should reject if key not found in results', async () => {
      const batchFn = vi.fn().mockResolvedValue(new Map([[1, 'result']]));

      const batcher = createBatcher<number, string>({ batchFn });

      const promise1 = batcher.request(1);
      const promise2 = batcher.request(2); // Not in results

      await vi.advanceTimersByTimeAsync(20);

      await expect(promise1).resolves.toBe('result');
      await expect(promise2).rejects.toThrow('No result for key');
    });
  });

  describe('flush', () => {
    it('should execute pending requests immediately', async () => {
      const batchFn = vi.fn().mockResolvedValue(
        new Map([
          [1, 'result1'],
          [2, 'result2'],
        ])
      );

      const batcher = createBatcher<number, string>({
        batchFn,
        maxWait: 1000,
      });

      const promise1 = batcher.request(1);
      const promise2 = batcher.request(2);

      expect(batchFn).not.toHaveBeenCalled();

      batcher.flush();

      await vi.advanceTimersByTimeAsync(0);

      expect(batchFn).toHaveBeenCalledTimes(1);
      expect(await promise1).toBe('result1');
      expect(await promise2).toBe('result2');
    });
  });

  describe('cancel', () => {
    it('should reject all pending requests', async () => {
      const batchFn = vi.fn().mockResolvedValue(new Map());

      const batcher = createBatcher<number, string>({
        batchFn,
        maxWait: 1000,
      });

      const promise1 = batcher.request(1);
      const promise2 = batcher.request(2);

      batcher.cancel();

      await expect(promise1).rejects.toThrow('cancelled');
      await expect(promise2).rejects.toThrow('cancelled');
      expect(batchFn).not.toHaveBeenCalled();
    });
  });

  describe('getPendingCount', () => {
    it('should return number of pending requests', () => {
      const batchFn = vi.fn().mockResolvedValue(new Map());

      const batcher = createBatcher<number, string>({ batchFn });

      expect(batcher.getPendingCount()).toBe(0);

      batcher.request(1);
      expect(batcher.getPendingCount()).toBe(1);

      batcher.request(2);
      expect(batcher.getPendingCount()).toBe(2);
    });
  });
});

describe('createPredictionBatcher', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should create a batcher with 50ms maxWait', async () => {
    const batchFn = vi.fn().mockResolvedValue(
      new Map([
        ['1h', { price: 0.001 }],
        ['4h', { price: 0.002 }],
      ])
    );

    const batcher = createPredictionBatcher(batchFn);

    batcher.request('1h');
    batcher.request('4h');

    await vi.advanceTimersByTimeAsync(40);
    expect(batchFn).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(15);
    expect(batchFn).toHaveBeenCalledTimes(1);
  });
});

describe('createChainBatcher', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should create a batcher with 30ms maxWait', async () => {
    const batchFn = vi.fn().mockResolvedValue(
      new Map([
        [1, { name: 'Ethereum' }],
        [8453, { name: 'Base' }],
      ])
    );

    const batcher = createChainBatcher(batchFn);

    batcher.request(1);
    batcher.request(8453);

    await vi.advanceTimersByTimeAsync(25);
    expect(batchFn).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(10);
    expect(batchFn).toHaveBeenCalledTimes(1);
  });
});

describe('RateLimitedQueue', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('should process requests with minimum delay', async () => {
    const queue = new RateLimitedQueue<number>(100);
    const results: number[] = [];

    const promise1 = queue.enqueue(async () => {
      results.push(1);
      return 1;
    });
    const promise2 = queue.enqueue(async () => {
      results.push(2);
      return 2;
    });
    const promise3 = queue.enqueue(async () => {
      results.push(3);
      return 3;
    });

    // First request executes immediately
    await vi.advanceTimersByTimeAsync(0);
    expect(results).toEqual([1]);

    // Wait for delay between requests
    await vi.advanceTimersByTimeAsync(100);
    expect(results).toEqual([1, 2]);

    await vi.advanceTimersByTimeAsync(100);
    expect(results).toEqual([1, 2, 3]);

    expect(await promise1).toBe(1);
    expect(await promise2).toBe(2);
    expect(await promise3).toBe(3);
  });

  it('should track pending count', () => {
    const queue = new RateLimitedQueue<number>();

    expect(queue.pendingCount).toBe(0);

    queue.enqueue(async () => 1);
    queue.enqueue(async () => 2);

    // First one starts processing immediately, second is pending
    expect(queue.pendingCount).toBe(1);
  });

  it('should handle errors without stopping queue', async () => {
    const queue = new RateLimitedQueue<number>(50);

    const promise1 = queue.enqueue(async () => {
      throw new Error('Error 1');
    });

    const promise2 = queue.enqueue(async () => {
      return 2;
    });

    await vi.advanceTimersByTimeAsync(0);
    await expect(promise1).rejects.toThrow('Error 1');

    await vi.advanceTimersByTimeAsync(50);
    await expect(promise2).resolves.toBe(2);
  });

  it('should clear pending requests', () => {
    const queue = new RateLimitedQueue<number>();

    queue.enqueue(async () => 1);
    queue.enqueue(async () => 2);
    queue.enqueue(async () => 3);

    queue.clear();

    expect(queue.pendingCount).toBe(0);
  });
});
