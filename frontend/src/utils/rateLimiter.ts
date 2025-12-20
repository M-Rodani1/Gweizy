/**
 * Client-side rate limiter
 * Prevents excessive API calls from the frontend
 */

import { RATE_LIMIT } from '../constants';

interface RequestRecord {
  count: number;
  resetTime: number;
}

class RateLimiter {
  private requests: Map<string, RequestRecord> = new Map();

  /**
   * Check if a request is allowed
   * @param key - Unique identifier for the rate limit (e.g., endpoint name)
   * @param maxRequests - Maximum requests allowed
   * @param windowMs - Time window in milliseconds
   * @returns true if request is allowed, false if rate limited
   */
  isAllowed(key: string, maxRequests: number = RATE_LIMIT.MAX_REQUESTS_PER_MINUTE, windowMs: number = RATE_LIMIT.WINDOW_MS): boolean {
    const now = Date.now();
    const record = this.requests.get(key);

    if (!record || now > record.resetTime) {
      // Create new record or reset expired record
      this.requests.set(key, {
        count: 1,
        resetTime: now + windowMs
      });
      return true;
    }

    if (record.count >= maxRequests) {
      return false;
    }

    record.count++;
    return true;
  }

  /**
   * Get time until rate limit resets
   * @param key - Rate limit key
   * @returns Milliseconds until reset, or 0 if not rate limited
   */
  getTimeUntilReset(key: string): number {
    const record = this.requests.get(key);
    if (!record) return 0;
    
    const now = Date.now();
    return Math.max(0, record.resetTime - now);
  }

  /**
   * Clear all rate limit records
   */
  clear(): void {
    this.requests.clear();
  }

  /**
   * Remove expired records (cleanup)
   */
  cleanup(): void {
    const now = Date.now();
    for (const [key, record] of this.requests.entries()) {
      if (now > record.resetTime) {
        this.requests.delete(key);
      }
    }
  }
}

// Singleton instance
export const rateLimiter = new RateLimiter();

// Cleanup expired records every minute
if (typeof window !== 'undefined') {
  setInterval(() => {
    rateLimiter.cleanup();
  }, 60000);
}
