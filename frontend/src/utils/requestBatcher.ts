/**
 * Request batching utility for optimizing API calls.
 *
 * Collects multiple requests over a short window and combines them
 * into a single API call, reducing network overhead and server load.
 *
 * @module utils/requestBatcher
 */

/**
 * Configuration options for the request batcher.
 */
export interface BatcherOptions<K, V> {
  /** Function to execute the batched request */
  batchFn: (keys: K[]) => Promise<Map<K, V>>;
  /** Maximum time to wait before executing batch (default: 16ms) */
  maxWait?: number;
  /** Maximum batch size before forcing execution (default: 100) */
  maxBatchSize?: number;
  /** Custom key serializer for deduplication (default: JSON.stringify) */
  keySerializer?: (key: K) => string;
}

/**
 * Pending request entry.
 */
interface PendingRequest<K, V> {
  key: K;
  resolve: (value: V) => void;
  reject: (error: Error) => void;
}

/**
 * Creates a request batcher that collects individual requests
 * and executes them in batches.
 *
 * @param options - Batcher configuration
 * @returns Function to request individual items
 *
 * @example
 * ```ts
 * // Create a batcher for fetching predictions by chain ID
 * const fetchPredictions = createBatcher<number, PredictionData>({
 *   batchFn: async (chainIds) => {
 *     const results = await api.fetchPredictions(chainIds);
 *     return new Map(results.map(r => [r.chainId, r]));
 *   },
 *   maxWait: 50,
 *   maxBatchSize: 10,
 * });
 *
 * // Individual calls are automatically batched
 * const [eth, base, arb] = await Promise.all([
 *   fetchPredictions(1),    // Ethereum
 *   fetchPredictions(8453), // Base
 *   fetchPredictions(42161) // Arbitrum
 * ]);
 * ```
 */
export function createBatcher<K, V>(options: BatcherOptions<K, V>) {
  const {
    batchFn,
    maxWait = 16, // ~1 frame at 60fps
    maxBatchSize = 100,
    keySerializer = (key: K) => JSON.stringify(key),
  } = options;

  let pending: PendingRequest<K, V>[] = [];
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let scheduledKeys = new Set<string>();

  const executeBatch = async () => {
    if (pending.length === 0) return;

    const batch = pending;
    pending = [];
    timeoutId = null;
    scheduledKeys.clear();

    // Extract unique keys for the batch request
    const keys = batch.map((item) => item.key);

    try {
      const results = await batchFn(keys);

      // Resolve each pending request with its result
      for (const request of batch) {
        const result = results.get(request.key);
        if (result !== undefined) {
          request.resolve(result);
        } else {
          request.reject(new Error(`No result for key: ${keySerializer(request.key)}`));
        }
      }
    } catch (error) {
      // Reject all pending requests on batch failure
      const err = error instanceof Error ? error : new Error(String(error));
      for (const request of batch) {
        request.reject(err);
      }
    }
  };

  const scheduleBatch = () => {
    if (timeoutId === null) {
      timeoutId = setTimeout(executeBatch, maxWait);
    }
  };

  /**
   * Request a single item. The request will be batched with others.
   */
  const request = (key: K): Promise<V> => {
    return new Promise((resolve, reject) => {
      const serializedKey = keySerializer(key);

      // Deduplicate requests for the same key
      if (!scheduledKeys.has(serializedKey)) {
        scheduledKeys.add(serializedKey);
        pending.push({ key, resolve, reject });
      } else {
        // Find existing request and add to its callbacks
        const existing = pending.find((p) => keySerializer(p.key) === serializedKey);
        if (existing) {
          const originalResolve = existing.resolve;
          const originalReject = existing.reject;
          existing.resolve = (value: V) => {
            originalResolve(value);
            resolve(value);
          };
          existing.reject = (error: Error) => {
            originalReject(error);
            reject(error);
          };
        }
      }

      scheduleBatch();

      // Force execution if batch is full
      if (pending.length >= maxBatchSize) {
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        executeBatch();
      }
    });
  };

  /**
   * Flush any pending requests immediately.
   */
  const flush = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    executeBatch();
  };

  /**
   * Cancel all pending requests.
   */
  const cancel = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    const err = new Error('Batch request cancelled');
    for (const request of pending) {
      request.reject(err);
    }
    pending = [];
    scheduledKeys.clear();
  };

  /**
   * Get number of pending requests.
   */
  const getPendingCount = () => pending.length;

  return {
    request,
    flush,
    cancel,
    getPendingCount,
  };
}

/**
 * Creates a prediction batcher for fetching predictions by time horizon.
 *
 * @example
 * ```ts
 * const batcher = createPredictionBatcher(async (horizons) => {
 *   const response = await fetchPredictions({ horizons });
 *   return new Map(horizons.map(h => [h, response.predictions[h]]));
 * });
 *
 * // Batch multiple prediction requests
 * const [h1, h4, h24] = await Promise.all([
 *   batcher.request('1h'),
 *   batcher.request('4h'),
 *   batcher.request('24h'),
 * ]);
 * ```
 */
export function createPredictionBatcher<V>(
  batchFn: (horizons: string[]) => Promise<Map<string, V>>
) {
  return createBatcher<string, V>({
    batchFn,
    maxWait: 50, // 50ms window for collecting prediction requests
    maxBatchSize: 10,
  });
}

/**
 * Creates a chain data batcher for fetching data by chain ID.
 *
 * @example
 * ```ts
 * const batcher = createChainBatcher(async (chainIds) => {
 *   const results = await Promise.all(
 *     chainIds.map(id => fetchChainData(id))
 *   );
 *   return new Map(chainIds.map((id, i) => [id, results[i]]));
 * });
 *
 * // Batch multiple chain requests
 * const [eth, base] = await Promise.all([
 *   batcher.request(1),
 *   batcher.request(8453),
 * ]);
 * ```
 */
export function createChainBatcher<V>(
  batchFn: (chainIds: number[]) => Promise<Map<number, V>>
) {
  return createBatcher<number, V>({
    batchFn,
    maxWait: 30, // 30ms window for chain data requests
    maxBatchSize: 20,
  });
}

/**
 * Rate-limited request queue that processes requests sequentially
 * with a minimum delay between each.
 */
export class RateLimitedQueue<T> {
  private queue: Array<() => Promise<T>> = [];
  private prioritizedQueue: Array<() => Promise<T>> = [];
  private processing = false;
  private scheduled = false;
  private minDelay: number;

  constructor(minDelayMs = 100) {
    this.minDelay = minDelayMs;
  }

  /**
   * Add a request to the queue.
   */
  async enqueue(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
        try {
          const result = await fn();
          resolve(result);
          return result;
        } catch (error) {
          reject(error);
          throw error;
        }
      });

      this.scheduleProcess();
    });
  }

  /**
   * Add a high-priority request to the front of the queue.
   */
  async enqueuePriority(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.prioritizedQueue.push(async () => {
        try {
          const result = await fn();
          resolve(result);
          return result;
        } catch (error) {
          reject(error);
          throw error;
        }
      });

      this.scheduleProcess();
    });
  }

  private scheduleProcess() {
    if (this.processing || this.scheduled) return;
    this.scheduled = true;
    const queueMicrotaskFn = (globalThis as { queueMicrotask?: (cb: () => void) => void }).queueMicrotask;
    if (queueMicrotaskFn) {
      queueMicrotaskFn(() => {
        this.scheduled = false;
        this.processQueue();
      });
    } else {
      Promise.resolve().then(() => {
        this.scheduled = false;
        this.processQueue();
      });
    }
  }

  private async processQueue() {
    if (this.processing || (this.queue.length === 0 && this.prioritizedQueue.length === 0)) return;

    this.processing = true;

    while (this.prioritizedQueue.length > 0 || this.queue.length > 0) {
      const fn = this.prioritizedQueue.shift() ?? this.queue.shift();
      if (fn) {
        try {
          await fn();
        } catch {
          // Error already handled by promise rejection
        }

        if ((this.prioritizedQueue.length > 0 || this.queue.length > 0) && this.minDelay > 0) {
          await new Promise((resolve) => setTimeout(resolve, this.minDelay));
        }
      }
    }

    this.processing = false;
  }

  /**
   * Clear all pending requests.
   */
  clear() {
    this.queue = [];
    this.prioritizedQueue = [];
  }

  /**
   * Get number of pending requests.
   */
  get pendingCount() {
    return this.queue.length + this.prioritizedQueue.length;
  }
}

/**
 * Priority request queue that processes higher priority items first.
 */
export class PriorityRequestQueue<T> {
  private queues: Map<number, Array<() => Promise<T>>> = new Map();
  private processing = false;
  private scheduled = false;
  private minDelay: number;

  constructor(minDelayMs = 0) {
    this.minDelay = minDelayMs;
  }

  async enqueue(fn: () => Promise<T>, priority = 0): Promise<T> {
    return new Promise((resolve, reject) => {
      const queue = this.getQueue(priority);
      queue.push(async () => {
        try {
          const result = await fn();
          resolve(result);
          return result;
        } catch (error) {
          reject(error);
          throw error;
        }
      });

      this.scheduleProcess();
    });
  }

  private getQueue(priority: number) {
    if (!this.queues.has(priority)) {
      this.queues.set(priority, []);
    }
    return this.queues.get(priority) as Array<() => Promise<T>>;
  }

  private async processQueue() {
    if (this.processing) return;
    this.processing = true;

    while (this.pendingCount > 0) {
      const priorities = Array.from(this.queues.keys()).sort((a, b) => b - a);
      let task: (() => Promise<T>) | undefined;

      for (const priority of priorities) {
        const queue = this.queues.get(priority);
        if (queue && queue.length > 0) {
          task = queue.shift();
          break;
        }
      }

      if (task) {
        try {
          await task();
        } catch {
          // Error already handled by promise rejection
        }

        if (this.pendingCount > 0 && this.minDelay > 0) {
          await new Promise((resolve) => setTimeout(resolve, this.minDelay));
        }
      }
    }

    this.processing = false;
  }

  private scheduleProcess() {
    if (this.processing || this.scheduled) return;
    this.scheduled = true;
    const queueMicrotaskFn = (globalThis as { queueMicrotask?: (cb: () => void) => void }).queueMicrotask;
    if (queueMicrotaskFn) {
      queueMicrotaskFn(() => {
        this.scheduled = false;
        this.processQueue();
      });
    } else {
      Promise.resolve().then(() => {
        this.scheduled = false;
        this.processQueue();
      });
    }
  }

  clear() {
    this.queues.clear();
  }

  get pendingCount() {
    let total = 0;
    for (const queue of this.queues.values()) {
      total += queue.length;
    }
    return total;
  }
}

export default createBatcher;
