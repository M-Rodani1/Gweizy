/**
 * Rate limiting utilities for API calls.
 *
 * Provides client-side rate limiting awareness to prevent
 * hitting API rate limits and handle 429 responses gracefully.
 *
 * @module utils/rateLimit
 */

import { API_CONFIG } from '../config/api';
import { withTimeout } from './withTimeout';

/**
 * Rate limit state for an endpoint.
 */
export interface RateLimitState {
  /** Number of requests remaining in the current window */
  remaining: number;
  /** Total requests allowed per window */
  limit: number;
  /** Timestamp when the rate limit resets */
  resetTime: number;
  /** Whether currently rate limited */
  isLimited: boolean;
}

/**
 * Rate limit headers from API response.
 */
export interface RateLimitHeaders {
  'x-ratelimit-limit'?: string;
  'x-ratelimit-remaining'?: string;
  'x-ratelimit-reset'?: string;
  'retry-after'?: string;
}

/**
 * Rate limiter configuration.
 */
export interface RateLimiterConfig {
  /** Maximum requests per window (default: 60) */
  maxRequests?: number;
  /** Window duration in milliseconds (default: 60000 = 1 minute) */
  windowMs?: number;
  /** Callback when rate limited */
  onRateLimited?: (retryAfter: number) => void;
}

/**
 * Client-side rate limiter that tracks request counts
 * and prevents exceeding rate limits.
 */
export class RateLimiter {
  private maxRequests: number;
  private windowMs: number;
  private requests: number[] = [];
  private serverState: RateLimitState | null = null;
  private onRateLimited?: (retryAfter: number) => void;

  constructor(config: RateLimiterConfig = {}) {
    this.maxRequests = config.maxRequests ?? 60;
    this.windowMs = config.windowMs ?? 60000;
    this.onRateLimited = config.onRateLimited;
  }

  /**
   * Check if a request can be made without hitting rate limits.
   *
   * @returns True if request is allowed
   */
  canMakeRequest(): boolean {
    this.pruneOldRequests();

    // Check server-reported rate limit first
    if (this.serverState?.isLimited) {
      if (Date.now() < this.serverState.resetTime) {
        return false;
      }
      // Reset after window expires
      this.serverState.isLimited = false;
    }

    // Check client-side tracking
    return this.requests.length < this.maxRequests;
  }

  /**
   * Record a request and return whether it was allowed.
   *
   * @returns True if request was recorded, false if rate limited
   */
  recordRequest(): boolean {
    if (!this.canMakeRequest()) {
      const retryAfter = this.getRetryAfter();
      this.onRateLimited?.(retryAfter);
      return false;
    }

    this.requests.push(Date.now());
    return true;
  }

  /**
   * Update rate limit state from response headers.
   *
   * @param headers - Response headers object or Headers instance
   */
  updateFromHeaders(headers: Headers | RateLimitHeaders | Record<string, string>): void {
    const getHeader = (name: string): string | null => {
      if (headers instanceof Headers) {
        return headers.get(name);
      }
      const key = name.toLowerCase();
      return (headers as Record<string, string>)[key] ?? (headers as Record<string, string>)[name] ?? null;
    };

    const limit = getHeader('x-ratelimit-limit');
    const remaining = getHeader('x-ratelimit-remaining');
    const reset = getHeader('x-ratelimit-reset');
    const retryAfter = getHeader('retry-after');

    if (limit || remaining || reset) {
      this.serverState = {
        limit: limit ? parseInt(limit, 10) : this.maxRequests,
        remaining: remaining ? parseInt(remaining, 10) : this.maxRequests,
        resetTime: reset ? parseInt(reset, 10) * 1000 : Date.now() + this.windowMs,
        isLimited: remaining === '0',
      };

      // Sync client limit with server
      if (limit) {
        this.maxRequests = parseInt(limit, 10);
      }
    }

    if (retryAfter) {
      const retryMs = parseInt(retryAfter, 10) * 1000;
      this.serverState = {
        limit: this.serverState?.limit ?? this.maxRequests,
        remaining: 0,
        resetTime: Date.now() + retryMs,
        isLimited: true,
      };
    }
  }

  /**
   * Handle a 429 response.
   *
   * @param retryAfterSeconds - Seconds to wait before retrying
   */
  handleRateLimitResponse(retryAfterSeconds = 60): void {
    this.serverState = {
      limit: this.serverState?.limit ?? this.maxRequests,
      remaining: 0,
      resetTime: Date.now() + retryAfterSeconds * 1000,
      isLimited: true,
    };

    this.onRateLimited?.(retryAfterSeconds * 1000);
  }

  /**
   * Get the time in milliseconds to wait before retrying.
   *
   * @returns Milliseconds to wait, or 0 if not rate limited
   */
  getRetryAfter(): number {
    if (this.serverState?.isLimited) {
      return Math.max(0, this.serverState.resetTime - Date.now());
    }

    if (this.requests.length >= this.maxRequests) {
      const oldestRequest = this.requests[0];
      return Math.max(0, oldestRequest + this.windowMs - Date.now());
    }

    return 0;
  }

  /**
   * Get current rate limit state.
   *
   * @returns Current state
   */
  getState(): RateLimitState {
    this.pruneOldRequests();

    return {
      limit: this.serverState?.limit ?? this.maxRequests,
      remaining: this.serverState?.remaining ?? this.maxRequests - this.requests.length,
      resetTime: this.serverState?.resetTime ?? Date.now() + this.windowMs,
      isLimited: !this.canMakeRequest(),
    };
  }

  /**
   * Reset the rate limiter state.
   */
  reset(): void {
    this.requests = [];
    this.serverState = null;
  }

  /**
   * Remove requests outside the current window.
   */
  private pruneOldRequests(): void {
    const cutoff = Date.now() - this.windowMs;
    this.requests = this.requests.filter((time) => time > cutoff);
  }
}

/**
 * Create a fetch wrapper that handles rate limiting.
 *
 * @param rateLimiter - RateLimiter instance
 * @returns Wrapped fetch function
 *
 * @example
 * ```ts
 * const limiter = new RateLimiter({ maxRequests: 30 });
 * const rateLimitedFetch = createRateLimitedFetch(limiter);
 *
 * const response = await rateLimitedFetch('/api/data');
 * ```
 */
export function createRateLimitedFetch(rateLimiter: RateLimiter) {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    // Check if we can make a request
    if (!rateLimiter.canMakeRequest()) {
      const retryAfter = rateLimiter.getRetryAfter();
      throw new RateLimitError(
        `Rate limited. Retry after ${Math.ceil(retryAfter / 1000)} seconds`,
        retryAfter
      );
    }

    rateLimiter.recordRequest();

    const response = await withTimeout(
      fetch(input, init),
      API_CONFIG.TIMEOUT,
      'Request timed out: rate-limited fetch'
    );

    // Update state from response headers
    rateLimiter.updateFromHeaders(response.headers);

    // Handle 429 response
    if (response.status === 429) {
      const retryAfter = response.headers.get('retry-after');
      rateLimiter.handleRateLimitResponse(retryAfter ? parseInt(retryAfter, 10) : 60);
      throw new RateLimitError(
        'Too Many Requests',
        rateLimiter.getRetryAfter()
      );
    }

    return response;
  };
}

/**
 * Error thrown when rate limited.
 */
export class RateLimitError extends Error {
  readonly retryAfter: number;
  readonly isRateLimitError = true;

  constructor(message: string, retryAfter: number) {
    super(message);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

/**
 * Check if an error is a rate limit error.
 */
export function isRateLimitError(error: unknown): error is RateLimitError {
  return error instanceof RateLimitError || (error as RateLimitError)?.isRateLimitError === true;
}

/**
 * Format retry time for display.
 *
 * @param ms - Milliseconds to wait
 * @returns Human-readable string
 */
export function formatRetryTime(ms: number): string {
  if (ms < 1000) return 'less than a second';
  if (ms < 60000) return `${Math.ceil(ms / 1000)} seconds`;
  if (ms < 3600000) return `${Math.ceil(ms / 60000)} minutes`;
  return `${Math.ceil(ms / 3600000)} hours`;
}

// Default rate limiter instance
export const defaultRateLimiter = new RateLimiter();

export default RateLimiter;
