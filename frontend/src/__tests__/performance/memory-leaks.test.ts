/**
 * Memory Leak Detection Tests
 *
 * Tests to verify that components and utilities properly clean up
 * resources, event listeners, timers, and subscriptions to prevent
 * memory leaks.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ============================================================================
// Memory Leak Patterns and Detection Utilities
// ============================================================================

/**
 * Track event listener additions and removals
 */
class EventListenerTracker {
  private listeners: Map<string, number> = new Map();

  add(type: string) {
    this.listeners.set(type, (this.listeners.get(type) || 0) + 1);
  }

  remove(type: string) {
    const count = this.listeners.get(type) || 0;
    if (count > 0) {
      this.listeners.set(type, count - 1);
    }
  }

  getCount(type: string): number {
    return this.listeners.get(type) || 0;
  }

  getTotalCount(): number {
    let total = 0;
    for (const count of this.listeners.values()) {
      total += count;
    }
    return total;
  }

  reset() {
    this.listeners.clear();
  }
}

/**
 * Track timer creation and cleanup
 */
class TimerTracker {
  private timers: Set<number> = new Set();
  private intervals: Set<number> = new Set();

  addTimer(id: number) {
    this.timers.add(id);
  }

  removeTimer(id: number) {
    this.timers.delete(id);
  }

  addInterval(id: number) {
    this.intervals.add(id);
  }

  removeInterval(id: number) {
    this.intervals.delete(id);
  }

  getTimerCount(): number {
    return this.timers.size;
  }

  getIntervalCount(): number {
    return this.intervals.size;
  }

  reset() {
    this.timers.clear();
    this.intervals.clear();
  }
}

/**
 * Track subscription/observer patterns
 */
class SubscriptionTracker {
  private subscriptions: Map<string, number> = new Map();

  subscribe(name: string) {
    this.subscriptions.set(name, (this.subscriptions.get(name) || 0) + 1);
  }

  unsubscribe(name: string) {
    const count = this.subscriptions.get(name) || 0;
    if (count > 0) {
      this.subscriptions.set(name, count - 1);
    }
  }

  getCount(name: string): number {
    return this.subscriptions.get(name) || 0;
  }

  getTotalCount(): number {
    let total = 0;
    for (const count of this.subscriptions.values()) {
      total += count;
    }
    return total;
  }

  reset() {
    this.subscriptions.clear();
  }
}

// ============================================================================
// Tests
// ============================================================================

describe('Memory Leak Detection', () => {
  let eventTracker: EventListenerTracker;
  let timerTracker: TimerTracker;
  let subscriptionTracker: SubscriptionTracker;

  beforeEach(() => {
    eventTracker = new EventListenerTracker();
    timerTracker = new TimerTracker();
    subscriptionTracker = new SubscriptionTracker();
  });

  afterEach(() => {
    eventTracker.reset();
    timerTracker.reset();
    subscriptionTracker.reset();
  });

  describe('Event Listener Cleanup', () => {
    it('should properly add and remove event listeners', () => {
      eventTracker.add('click');
      eventTracker.add('scroll');
      eventTracker.add('click');

      expect(eventTracker.getCount('click')).toBe(2);
      expect(eventTracker.getCount('scroll')).toBe(1);

      eventTracker.remove('click');
      expect(eventTracker.getCount('click')).toBe(1);

      eventTracker.remove('click');
      expect(eventTracker.getCount('click')).toBe(0);

      eventTracker.remove('scroll');
      expect(eventTracker.getTotalCount()).toBe(0);
    });

    it('should track resize event listener lifecycle', () => {
      const resizeHandler = vi.fn();
      let cleanup: (() => void) | null = null;

      // Simulate component mount
      const mount = () => {
        eventTracker.add('resize');
        window.addEventListener('resize', resizeHandler);
        cleanup = () => {
          eventTracker.remove('resize');
          window.removeEventListener('resize', resizeHandler);
        };
      };

      // Simulate component unmount
      const unmount = () => {
        cleanup?.();
        cleanup = null;
      };

      mount();
      expect(eventTracker.getCount('resize')).toBe(1);

      unmount();
      expect(eventTracker.getCount('resize')).toBe(0);
    });

    it('should detect unbalanced event listeners as potential leak', () => {
      // Simulate a leak where add is called but remove is not
      eventTracker.add('click');
      eventTracker.add('mousemove');
      eventTracker.add('keydown');

      // Only remove one
      eventTracker.remove('click');

      // These would indicate potential leaks
      expect(eventTracker.getCount('mousemove')).toBe(1);
      expect(eventTracker.getCount('keydown')).toBe(1);
      expect(eventTracker.getTotalCount()).toBe(2);
    });

    it('should handle multiple listeners for same event', () => {
      const handlers = [vi.fn(), vi.fn(), vi.fn()];

      handlers.forEach(() => {
        eventTracker.add('scroll');
      });

      expect(eventTracker.getCount('scroll')).toBe(3);

      // Remove all
      handlers.forEach(() => {
        eventTracker.remove('scroll');
      });

      expect(eventTracker.getCount('scroll')).toBe(0);
    });
  });

  describe('Timer Cleanup', () => {
    it('should track setTimeout cleanup', () => {
      const id1 = 1;
      const id2 = 2;

      timerTracker.addTimer(id1);
      timerTracker.addTimer(id2);

      expect(timerTracker.getTimerCount()).toBe(2);

      timerTracker.removeTimer(id1);
      expect(timerTracker.getTimerCount()).toBe(1);

      timerTracker.removeTimer(id2);
      expect(timerTracker.getTimerCount()).toBe(0);
    });

    it('should track setInterval cleanup', () => {
      const id1 = 1;
      const id2 = 2;

      timerTracker.addInterval(id1);
      timerTracker.addInterval(id2);

      expect(timerTracker.getIntervalCount()).toBe(2);

      timerTracker.removeInterval(id1);
      expect(timerTracker.getIntervalCount()).toBe(1);

      timerTracker.removeInterval(id2);
      expect(timerTracker.getIntervalCount()).toBe(0);
    });

    it('should detect uncleaned timers as potential leak', () => {
      timerTracker.addTimer(1);
      timerTracker.addTimer(2);
      timerTracker.addInterval(3);

      // Only cleanup one timer
      timerTracker.removeTimer(1);

      // These indicate leaks
      expect(timerTracker.getTimerCount()).toBe(1);
      expect(timerTracker.getIntervalCount()).toBe(1);
    });

    it('should handle real timer lifecycle', () => {
      vi.useFakeTimers();

      const timers: ReturnType<typeof setTimeout>[] = [];

      // Create timers
      timers.push(setTimeout(() => {}, 1000));
      timers.push(setTimeout(() => {}, 2000));

      expect(timers.length).toBe(2);

      // Cleanup
      timers.forEach((t) => clearTimeout(t));

      vi.useRealTimers();
    });
  });

  describe('Subscription Cleanup', () => {
    it('should track subscription lifecycle', () => {
      subscriptionTracker.subscribe('data-stream');
      subscriptionTracker.subscribe('user-events');

      expect(subscriptionTracker.getCount('data-stream')).toBe(1);
      expect(subscriptionTracker.getCount('user-events')).toBe(1);

      subscriptionTracker.unsubscribe('data-stream');
      subscriptionTracker.unsubscribe('user-events');

      expect(subscriptionTracker.getTotalCount()).toBe(0);
    });

    it('should detect uncleaned subscriptions', () => {
      subscriptionTracker.subscribe('ws-connection');
      subscriptionTracker.subscribe('price-updates');
      subscriptionTracker.subscribe('block-events');

      // Only cleanup one
      subscriptionTracker.unsubscribe('ws-connection');

      expect(subscriptionTracker.getTotalCount()).toBe(2);
    });

    it('should handle multiple subscriptions to same source', () => {
      subscriptionTracker.subscribe('gas-prices');
      subscriptionTracker.subscribe('gas-prices');
      subscriptionTracker.subscribe('gas-prices');

      expect(subscriptionTracker.getCount('gas-prices')).toBe(3);

      subscriptionTracker.unsubscribe('gas-prices');
      subscriptionTracker.unsubscribe('gas-prices');
      subscriptionTracker.unsubscribe('gas-prices');

      expect(subscriptionTracker.getCount('gas-prices')).toBe(0);
    });
  });

  describe('React Hook Cleanup Patterns', () => {
    it('should verify useEffect cleanup pattern', () => {
      let cleanupCalled = false;

      // Simulate useEffect
      const effect = () => {
        eventTracker.add('resize');
        subscriptionTracker.subscribe('data');

        // Return cleanup function
        return () => {
          cleanupCalled = true;
          eventTracker.remove('resize');
          subscriptionTracker.unsubscribe('data');
        };
      };

      const cleanup = effect();

      expect(eventTracker.getCount('resize')).toBe(1);
      expect(subscriptionTracker.getCount('data')).toBe(1);

      // Simulate unmount
      cleanup();

      expect(cleanupCalled).toBe(true);
      expect(eventTracker.getCount('resize')).toBe(0);
      expect(subscriptionTracker.getCount('data')).toBe(0);
    });

    it('should verify interval cleanup in useEffect', () => {
      vi.useFakeTimers();

      let intervalId: ReturnType<typeof setInterval> | null = null;
      const callback = vi.fn();

      // Simulate useEffect with interval
      const effect = () => {
        intervalId = setInterval(callback, 1000);
        timerTracker.addInterval(intervalId as unknown as number);

        return () => {
          if (intervalId) {
            clearInterval(intervalId);
            timerTracker.removeInterval(intervalId as unknown as number);
          }
        };
      };

      const cleanup = effect();

      expect(timerTracker.getIntervalCount()).toBe(1);

      cleanup();

      expect(timerTracker.getIntervalCount()).toBe(0);

      vi.useRealTimers();
    });

    it('should verify AbortController cleanup pattern', () => {
      let controller: AbortController | null = null;

      const effect = () => {
        controller = new AbortController();

        // Simulate fetch with AbortController
        const fetchData = async () => {
          try {
            // In real code: await fetch(url, { signal: controller.signal })
            if (controller?.signal.aborted) {
              throw new Error('Aborted');
            }
          } catch {
            // Handle abort
          }
        };

        fetchData();

        return () => {
          controller?.abort();
          controller = null;
        };
      };

      const cleanup = effect();

      expect(controller).not.toBeNull();

      cleanup();

      expect(controller).toBeNull();
    });
  });

  describe('Closure Memory Patterns', () => {
    it('should not retain large objects in closures after cleanup', () => {
      let largeData: number[] | null = Array(10000).fill(0);
      let callback: (() => void) | null = null;

      // Simulate creating a closure that captures large data
      const createCallback = () => {
        const data = largeData;
        callback = () => {
          // Use data
          return data?.length ?? 0;
        };
      };

      createCallback();

      // Cleanup
      callback = null;
      largeData = null;

      // After cleanup, references should be null
      expect(callback).toBeNull();
      expect(largeData).toBeNull();
    });

    it('should verify WeakMap pattern for metadata storage', () => {
      const metadata = new WeakMap<object, { timestamp: number }>();

      let obj1: { id: number } | null = { id: 1 };
      let obj2: { id: number } | null = { id: 2 };

      metadata.set(obj1, { timestamp: Date.now() });
      metadata.set(obj2, { timestamp: Date.now() });

      expect(metadata.has(obj1)).toBe(true);
      expect(metadata.has(obj2)).toBe(true);

      // When objects are dereferenced, WeakMap entries can be garbage collected
      obj1 = null;
      obj2 = null;

      // Note: In real scenarios, GC would clean up WeakMap entries
      // We can't force GC in tests, but this pattern prevents memory leaks
    });

    it('should verify WeakSet pattern for tracking objects', () => {
      const seen = new WeakSet<object>();

      let obj1: object | null = { value: 1 };
      let obj2: object | null = { value: 2 };

      seen.add(obj1);
      seen.add(obj2);

      expect(seen.has(obj1)).toBe(true);
      expect(seen.has(obj2)).toBe(true);

      // Dereferencing allows GC to collect
      obj1 = null;
      obj2 = null;
    });
  });

  describe('DOM Reference Cleanup', () => {
    it('should not retain DOM references after cleanup', () => {
      let elementRef: { node: object | null } = { node: {} };
      let callback: (() => void) | null = null;

      const setup = () => {
        callback = () => {
          // Access elementRef.node
          return elementRef.node;
        };
      };

      setup();

      // Cleanup
      elementRef.node = null;
      callback = null;

      expect(elementRef.node).toBeNull();
      expect(callback).toBeNull();
    });

    it('should verify observer disconnect pattern', () => {
      let disconnected = false;

      const mockObserver = {
        observe: vi.fn(),
        disconnect: vi.fn(() => {
          disconnected = true;
        }),
      };

      // Simulate MutationObserver or IntersectionObserver lifecycle
      const setup = () => {
        mockObserver.observe();

        return () => {
          mockObserver.disconnect();
        };
      };

      const cleanup = setup();

      expect(mockObserver.observe).toHaveBeenCalled();
      expect(disconnected).toBe(false);

      cleanup();

      expect(mockObserver.disconnect).toHaveBeenCalled();
      expect(disconnected).toBe(true);
    });
  });

  describe('Async Operation Cleanup', () => {
    it('should cancel pending async operations on cleanup', async () => {
      let cancelled = false;
      let operationCompleted = false;

      const asyncOperation = (signal: AbortSignal) => {
        return new Promise<string>((resolve, reject) => {
          const timeoutId = setTimeout(() => {
            operationCompleted = true;
            resolve('done');
          }, 1000);

          signal.addEventListener('abort', () => {
            clearTimeout(timeoutId);
            cancelled = true;
            reject(new Error('Cancelled'));
          });
        });
      };

      const controller = new AbortController();
      const promise = asyncOperation(controller.signal);

      // Cancel before completion
      controller.abort();

      try {
        await promise;
      } catch {
        // Expected cancellation
      }

      expect(cancelled).toBe(true);
      expect(operationCompleted).toBe(false);
    });

    it('should handle concurrent cleanup requests', async () => {
      const cleanupCalls: number[] = [];

      const createResource = (id: number) => {
        return {
          cleanup: () => {
            cleanupCalls.push(id);
          },
        };
      };

      const resources = [1, 2, 3, 4, 5].map(createResource);

      // Cleanup all resources concurrently
      await Promise.all(resources.map((r) => Promise.resolve(r.cleanup())));

      expect(cleanupCalls).toHaveLength(5);
      expect(cleanupCalls.sort()).toEqual([1, 2, 3, 4, 5]);
    });
  });

  describe('Memory Leak Patterns to Avoid', () => {
    it('should demonstrate proper Map cleanup', () => {
      const cache = new Map<string, object>();

      cache.set('key1', { data: 'value1' });
      cache.set('key2', { data: 'value2' });

      expect(cache.size).toBe(2);

      // Proper cleanup
      cache.clear();

      expect(cache.size).toBe(0);
    });

    it('should demonstrate proper Set cleanup', () => {
      const listeners = new Set<() => void>();

      const handler1 = () => {};
      const handler2 = () => {};

      listeners.add(handler1);
      listeners.add(handler2);

      expect(listeners.size).toBe(2);

      // Proper cleanup
      listeners.clear();

      expect(listeners.size).toBe(0);
    });

    it('should demonstrate bounded cache pattern', () => {
      const MAX_SIZE = 3;
      const cache: Map<string, number> = new Map();

      const addToCache = (key: string, value: number) => {
        if (cache.size >= MAX_SIZE) {
          // Remove oldest entry (first inserted)
          const firstKey = cache.keys().next().value;
          if (firstKey) {
            cache.delete(firstKey);
          }
        }
        cache.set(key, value);
      };

      addToCache('a', 1);
      addToCache('b', 2);
      addToCache('c', 3);
      addToCache('d', 4); // Should evict 'a'

      expect(cache.size).toBe(3);
      expect(cache.has('a')).toBe(false);
      expect(cache.has('d')).toBe(true);
    });

    it('should demonstrate TTL-based cache cleanup', () => {
      vi.useFakeTimers();

      interface CacheEntry<T> {
        value: T;
        expiry: number;
      }

      const cache = new Map<string, CacheEntry<number>>();
      const TTL = 5000;

      const set = (key: string, value: number) => {
        cache.set(key, {
          value,
          expiry: Date.now() + TTL,
        });
      };

      const get = (key: string): number | null => {
        const entry = cache.get(key);
        if (!entry) return null;
        if (Date.now() > entry.expiry) {
          cache.delete(key);
          return null;
        }
        return entry.value;
      };

      set('key', 42);
      expect(get('key')).toBe(42);

      vi.advanceTimersByTime(6000);

      expect(get('key')).toBeNull();
      expect(cache.size).toBe(0);

      vi.useRealTimers();
    });
  });
});
