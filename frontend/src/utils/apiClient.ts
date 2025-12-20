/**
 * Centralized API client with interceptors
 * Handles request/response logging, error handling, and rate limiting
 */

import { getApiUrl, API_CONFIG } from '../config/api';
import { rateLimiter } from './rateLimiter';
import { retryWithBackoff } from './retry';
import { getErrorMessage } from './errorMessages';
import toast from 'react-hot-toast';

/**
 * API client with interceptors
 */
class ApiClient {
  /**
   * Make a GET request
   */
  async get<T>(endpoint: string, params?: Record<string, string | number>): Promise<T> {
    const url = getApiUrl(endpoint, params);
    
    // Check rate limit
    if (!rateLimiter.isAllowed(endpoint)) {
      const timeUntilReset = rateLimiter.getTimeUntilReset(endpoint);
      throw new Error(`Rate limit exceeded. Please wait ${Math.ceil(timeUntilReset / 1000)} seconds`);
    }

    try {
      const response = await retryWithBackoff(
        async () => {
          const res = await fetch(url, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
            signal: AbortSignal.timeout(API_CONFIG.TIMEOUT),
          });

          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }

          return res;
        },
        {
          maxAttempts: API_CONFIG.RETRY_ATTEMPTS,
          baseDelay: API_CONFIG.RETRY_DELAY,
          onRetry: (attempt, error) => {
            console.warn(`Retrying ${endpoint} (attempt ${attempt}):`, error);
          }
        }
      );

      const data = await response.json();
      return data as T;
    } catch (error) {
      const errorInfo = getErrorMessage(error);
      
      // Only show toast for user-facing errors
      if (errorInfo.severity === 'error') {
        toast.error(errorInfo.message);
      }
      
      throw error;
    }
  }

  /**
   * Make a POST request
   */
  async post<T>(endpoint: string, body?: unknown): Promise<T> {
    const url = getApiUrl(endpoint);
    
    // Check rate limit
    if (!rateLimiter.isAllowed(endpoint)) {
      const timeUntilReset = rateLimiter.getTimeUntilReset(endpoint);
      throw new Error(`Rate limit exceeded. Please wait ${Math.ceil(timeUntilReset / 1000)} seconds`);
    }

    try {
      const response = await retryWithBackoff(
        async () => {
          const res = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: body ? JSON.stringify(body) : undefined,
            signal: AbortSignal.timeout(API_CONFIG.TIMEOUT),
          });

          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }

          return res;
        },
        {
          maxAttempts: API_CONFIG.RETRY_ATTEMPTS,
          baseDelay: API_CONFIG.RETRY_DELAY,
        }
      );

      const data = await response.json();
      return data as T;
    } catch (error) {
      const errorInfo = getErrorMessage(error);
      
      if (errorInfo.severity === 'error') {
        toast.error(errorInfo.message);
      }
      
      throw error;
    }
  }
}

// Singleton instance
export const apiClient = new ApiClient();
