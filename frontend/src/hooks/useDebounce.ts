/**
 * Debounce hook for optimizing input performance.
 *
 * Delays updating a value until after a specified delay has passed
 * since the last change. Useful for search inputs, form validation,
 * and reducing API calls.
 *
 * @module hooks/useDebounce
 */

import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Hook that returns a debounced version of the provided value.
 *
 * @param value - The value to debounce
 * @param delay - Delay in milliseconds (default: 300ms)
 * @returns The debounced value
 *
 * @example
 * ```tsx
 * function SearchComponent() {
 *   const [searchTerm, setSearchTerm] = useState('');
 *   const debouncedSearch = useDebounce(searchTerm, 500);
 *
 *   useEffect(() => {
 *     if (debouncedSearch) {
 *       // Only called 500ms after user stops typing
 *       performSearch(debouncedSearch);
 *     }
 *   }, [debouncedSearch]);
 *
 *   return (
 *     <input
 *       value={searchTerm}
 *       onChange={(e) => setSearchTerm(e.target.value)}
 *       placeholder="Search..."
 *     />
 *   );
 * }
 * ```
 */
export function useDebounce<T>(value: T, delay = 300): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}

/**
 * Configuration options for useDebouncedCallback.
 */
export interface DebouncedCallbackOptions {
  /** Delay in milliseconds (default: 300ms) */
  delay?: number;
  /** Execute on leading edge instead of trailing (default: false) */
  leading?: boolean;
  /** Maximum time to wait before forcing execution (default: undefined) */
  maxWait?: number;
}

/**
 * Hook that returns a debounced version of a callback function.
 *
 * @param callback - The function to debounce
 * @param options - Debounce configuration options
 * @returns Object with debounced function, cancel, and flush methods
 *
 * @example
 * ```tsx
 * function SearchComponent() {
 *   const [results, setResults] = useState([]);
 *
 *   const { debouncedCallback: search, cancel, isPending } = useDebouncedCallback(
 *     async (query: string) => {
 *       const results = await api.search(query);
 *       setResults(results);
 *     },
 *     { delay: 500 }
 *   );
 *
 *   return (
 *     <div>
 *       <input onChange={(e) => search(e.target.value)} />
 *       {isPending && <span>Searching...</span>}
 *       <button onClick={cancel}>Cancel</button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useDebouncedCallback<TArgs extends unknown[], TResult>(
  callback: (...args: TArgs) => TResult,
  options: DebouncedCallbackOptions = {}
): {
  debouncedCallback: (...args: TArgs) => void;
  cancel: () => void;
  flush: () => void;
  isPending: boolean;
} {
  const { delay = 300, leading = false, maxWait } = options;

  const [isPending, setIsPending] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const maxTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastArgsRef = useRef<TArgs | null>(null);
  const lastCallTimeRef = useRef<number | null>(null);
  const callbackRef = useRef(callback);

  // Keep callback ref updated
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  const cancel = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (maxTimeoutRef.current) {
      clearTimeout(maxTimeoutRef.current);
      maxTimeoutRef.current = null;
    }
    lastArgsRef.current = null;
    lastCallTimeRef.current = null;
    setIsPending(false);
  }, []);

  const flush = useCallback(() => {
    if (lastArgsRef.current && timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
      if (maxTimeoutRef.current) {
        clearTimeout(maxTimeoutRef.current);
        maxTimeoutRef.current = null;
      }
      callbackRef.current(...lastArgsRef.current);
      lastArgsRef.current = null;
      lastCallTimeRef.current = null;
      setIsPending(false);
    }
  }, []);

  const debouncedCallback = useCallback(
    (...args: TArgs) => {
      lastArgsRef.current = args;
      const now = Date.now();
      const isFirstCall = lastCallTimeRef.current === null;
      lastCallTimeRef.current = now;

      // Clear existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // Execute immediately on leading edge
      if (leading && isFirstCall) {
        callbackRef.current(...args);
        return;
      }

      setIsPending(true);

      // Set max wait timeout if configured
      if (maxWait && !maxTimeoutRef.current) {
        maxTimeoutRef.current = setTimeout(() => {
          if (lastArgsRef.current) {
            callbackRef.current(...lastArgsRef.current);
            lastArgsRef.current = null;
            lastCallTimeRef.current = null;
            setIsPending(false);
          }
          maxTimeoutRef.current = null;
        }, maxWait);
      }

      // Set regular debounce timeout
      timeoutRef.current = setTimeout(() => {
        if (lastArgsRef.current) {
          callbackRef.current(...lastArgsRef.current);
          lastArgsRef.current = null;
          lastCallTimeRef.current = null;
          setIsPending(false);
        }
        if (maxTimeoutRef.current) {
          clearTimeout(maxTimeoutRef.current);
          maxTimeoutRef.current = null;
        }
        timeoutRef.current = null;
      }, delay);
    },
    [delay, leading, maxWait]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (maxTimeoutRef.current) {
        clearTimeout(maxTimeoutRef.current);
      }
    };
  }, []);

  return { debouncedCallback, cancel, flush, isPending };
}

/**
 * Hook for debounced search input with built-in state management.
 *
 * @param initialValue - Initial search value (default: '')
 * @param delay - Debounce delay in milliseconds (default: 300ms)
 * @returns Object with value, debouncedValue, setValue, and clear methods
 *
 * @example
 * ```tsx
 * function SearchBar() {
 *   const { value, debouncedValue, setValue, clear } = useDebouncedSearch('', 500);
 *
 *   useEffect(() => {
 *     if (debouncedValue) {
 *       fetchSearchResults(debouncedValue);
 *     }
 *   }, [debouncedValue]);
 *
 *   return (
 *     <div>
 *       <input
 *         value={value}
 *         onChange={(e) => setValue(e.target.value)}
 *         placeholder="Search transactions..."
 *       />
 *       {value && <button onClick={clear}>Clear</button>}
 *     </div>
 *   );
 * }
 * ```
 */
export function useDebouncedSearch(initialValue = '', delay = 300) {
  const [value, setValue] = useState(initialValue);
  const debouncedValue = useDebounce(value, delay);

  const clear = useCallback(() => {
    setValue('');
  }, []);

  return {
    value,
    debouncedValue,
    setValue,
    clear,
    isSearching: value !== debouncedValue,
  };
}

export default useDebounce;
