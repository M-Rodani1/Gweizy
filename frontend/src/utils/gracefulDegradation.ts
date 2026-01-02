/**
 * Graceful degradation utilities
 * Provides fallback mechanisms when services fail
 */

import { getErrorMessage } from './errorMessages';

export interface FallbackData<T> {
  data: T;
  source: 'live' | 'cache' | 'fallback';
  timestamp: number;
}

/**
 * Try to fetch data with graceful degradation
 * Falls back to cached data or default values if fetch fails
 */
export async function fetchWithGracefulDegradation<T>(
  fetchFn: () => Promise<T>,
  fallbackFn?: () => Promise<T | null>,
  defaultValue?: T,
  cacheKey?: string
): Promise<FallbackData<T>> {
  // Try primary fetch
  try {
    const data = await fetchFn();
    // Store in cache if cache key provided
    if (cacheKey) {
      try {
        localStorage.setItem(cacheKey, JSON.stringify({
          data,
          timestamp: Date.now()
        }));
      } catch (e) {
        // Ignore localStorage errors
      }
    }
    return {
      data,
      source: 'live',
      timestamp: Date.now()
    };
  } catch (error) {
    console.warn('Primary fetch failed, trying fallback:', error);
    
    // Try fallback function
    if (fallbackFn) {
      try {
        const fallbackData = await fallbackFn();
        if (fallbackData !== null) {
          return {
            data: fallbackData,
            source: 'fallback',
            timestamp: Date.now()
          };
        }
      } catch (fallbackError) {
        console.warn('Fallback fetch also failed:', fallbackError);
      }
    }
    
    // Try cache
    if (cacheKey) {
      try {
        const cached = localStorage.getItem(cacheKey);
        if (cached) {
          const parsed = JSON.parse(cached);
          // Use cache if less than 5 minutes old
          if (Date.now() - parsed.timestamp < 5 * 60 * 1000) {
            return {
              data: parsed.data,
              source: 'cache',
              timestamp: parsed.timestamp
            };
          }
        }
      } catch (e) {
        // Ignore cache errors
      }
    }
    
    // Use default value
    if (defaultValue !== undefined) {
      const errorInfo = getErrorMessage(error);
      console.warn('Using default value due to error:', errorInfo.message);
      return {
        data: defaultValue,
        source: 'fallback',
        timestamp: Date.now()
      };
    }
    
    // Re-throw if no fallback available
    throw error;
  }
}

/**
 * Create a resilient fetch wrapper with multiple fallback strategies
 */
export function createResilientFetcher<T>(
  primaryUrl: string,
  fallbackUrls: string[] = [],
  defaultValue?: T
) {
  return async (): Promise<FallbackData<T>> => {
    const urls = [primaryUrl, ...fallbackUrls];
    
    for (let i = 0; i < urls.length; i++) {
      try {
        const response = await fetch(urls[i], {
          signal: AbortSignal.timeout(5000)
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        return {
          data,
          source: i === 0 ? 'live' : 'fallback',
          timestamp: Date.now()
        };
      } catch (error) {
        if (i === urls.length - 1) {
          // Last URL failed, use default or throw
          if (defaultValue !== undefined) {
            return {
              data: defaultValue,
              source: 'fallback',
              timestamp: Date.now()
            };
          }
          throw error;
        }
        // Try next URL
        continue;
      }
    }
    
    throw new Error('All fetch attempts failed');
  };
}

