/**
 * Request deduplication service
 * Prevents multiple identical requests from being made simultaneously
 */

interface PendingRequest<T> {
  promise: Promise<T>;
  timestamp: number;
}

class RequestDeduplicator {
  private pendingRequests: Map<string, PendingRequest<any>> = new Map();
  private readonly CACHE_DURATION = 1000; // 1 second

  /**
   * Deduplicate a request
   * @param key - Unique key for the request
   * @param requestFn - Function that makes the request
   * @returns Promise that resolves with the request result
   */
  async deduplicate<T>(key: string, requestFn: () => Promise<T>): Promise<T> {
    const existing = this.pendingRequests.get(key);
    
    // If request is pending and recent, return existing promise
    if (existing && Date.now() - existing.timestamp < this.CACHE_DURATION) {
      return existing.promise;
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
      timestamp: Date.now()
    });

    return promise;
  }

  /**
   * Clear all pending requests
   */
  clear(): void {
    this.pendingRequests.clear();
  }

  /**
   * Remove stale requests
   */
  cleanup(): void {
    const now = Date.now();
    for (const [key, request] of this.pendingRequests.entries()) {
      if (now - request.timestamp > this.CACHE_DURATION * 10) {
        this.pendingRequests.delete(key);
      }
    }
  }
}

// Singleton instance
export const requestDeduplicator = new RequestDeduplicator();

// Cleanup stale requests every 30 seconds
if (typeof window !== 'undefined') {
  setInterval(() => {
    requestDeduplicator.cleanup();
  }, 30000);
}
