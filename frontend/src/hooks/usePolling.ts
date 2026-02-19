/**
 * usePolling - A reusable hook for polling data at regular intervals
 *
 * Features:
 * - Automatic polling with configurable interval
 * - Immediate fetch on mount (optional)
 * - Pause/resume functionality
 * - Error handling and retry logic
 * - Cleanup on unmount
 */

import { useEffect, useRef, useCallback, useState } from 'react';

export interface UsePollingOptions<T> {
  /** The async function to call for fetching data */
  fetcher: () => Promise<T>;
  /** Polling interval in milliseconds (default: 30000 = 30s) */
  interval?: number;
  /** Whether to fetch immediately on mount (default: true) */
  immediate?: boolean;
  /** Whether polling is enabled (default: true) */
  enabled?: boolean;
  /** Callback when data is successfully fetched */
  onSuccess?: (data: T) => void;
  /** Callback when an error occurs */
  onError?: (error: Error) => void;
  /** Dependencies that trigger a re-fetch when changed */
  deps?: unknown[];
  /** Pause network fetches when document is hidden (default: true) */
  pauseWhenHidden?: boolean;
}

export interface UsePollingResult<T> {
  /** The fetched data */
  data: T | null;
  /** Whether a fetch is in progress */
  loading: boolean;
  /** Any error that occurred */
  error: Error | null;
  /** Manually trigger a fetch */
  refresh: () => Promise<void>;
  /** Stop polling */
  pause: () => void;
  /** Resume polling */
  resume: () => void;
  /** Whether polling is currently active */
  isPolling: boolean;
}

export function usePolling<T>({
  fetcher,
  interval = 30000,
  immediate = true,
  enabled = true,
  onSuccess,
  onError,
  deps = [],
  pauseWhenHidden = true
}: UsePollingOptions<T>): UsePollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(immediate);
  const [error, setError] = useState<Error | null>(null);
  const [isPolling, setIsPolling] = useState(enabled);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);
  const isDocumentVisible = useCallback(
    () => typeof document === 'undefined' || document.visibilityState === 'visible',
    []
  );

  const fetchData = useCallback(async () => {
    if (!mountedRef.current) return;

    try {
      setLoading(true);
      setError(null);
      const result = await fetcher();

      if (mountedRef.current) {
        setData(result);
        onSuccess?.(result);
      }
    } catch (err) {
      if (mountedRef.current) {
        const error = err instanceof Error ? err : new Error(String(err));
        setError(error);
        onError?.(error);
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [fetcher, onSuccess, onError]);

  const startPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    intervalRef.current = setInterval(() => {
      if (!pauseWhenHidden || isDocumentVisible()) {
        void fetchData();
      }
    }, interval);
  }, [fetchData, interval, isDocumentVisible, pauseWhenHidden]);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const pause = useCallback(() => {
    setIsPolling(false);
    stopPolling();
  }, [stopPolling]);

  const resume = useCallback(() => {
    setIsPolling(true);
    if (!pauseWhenHidden || isDocumentVisible()) {
      void fetchData();
    }
    startPolling();
  }, [fetchData, isDocumentVisible, pauseWhenHidden, startPolling]);

  const refresh = useCallback(async () => {
    await fetchData();
  }, [fetchData]);

  // Initial fetch and polling setup
  useEffect(() => {
    mountedRef.current = true;

    if (enabled && isPolling) {
      if (immediate && (!pauseWhenHidden || isDocumentVisible())) {
        void fetchData();
      }
      startPolling();
    }

    return () => {
      mountedRef.current = false;
      stopPolling();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, isDocumentVisible, pauseWhenHidden, ...deps]);

  // Handle enabled/isPolling changes
  useEffect(() => {
    if (enabled && isPolling) {
      startPolling();
    } else {
      stopPolling();
    }

    return () => stopPolling();
  }, [enabled, isPolling, startPolling, stopPolling]);

  useEffect(() => {
    if (!pauseWhenHidden) {
      return;
    }

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && enabled && isPolling) {
        void fetchData();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [enabled, fetchData, isPolling, pauseWhenHidden]);

  return {
    data,
    loading,
    error,
    refresh,
    pause,
    resume,
    isPolling
  };
}

export default usePolling;
