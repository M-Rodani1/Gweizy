/**
 * Hook for fetching and managing AI transaction recommendations
 */

import { useState, useEffect, useCallback } from 'react';
import { API_CONFIG, getApiUrl } from '../config/api';
import { REFRESH_INTERVALS } from '../constants';
import { withTimeout } from '../utils/withTimeout';

export interface AgentRecommendation {
  action: string;
  confidence: number;
  recommended_gas: number;
  expected_savings: number;
  reasoning: string;
  urgency_factor: number;
  wait_time?: number;
}

export type LoadingState = 'idle' | 'analyzing' | 'timeout' | 'error';

export interface UseRecommendationResult {
  recommendation: AgentRecommendation | null;
  loading: boolean;
  loadingState: LoadingState;
  error: string | null;
  retryCount: number;
  countdown: number | null;
  refresh: () => Promise<void>;
}

export const useRecommendation = (urgency: number): UseRecommendationResult => {
  const [recommendation, setRecommendation] = useState<AgentRecommendation | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingState, setLoadingState] = useState<LoadingState>('analyzing');
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [countdown, setCountdown] = useState<number | null>(null);

  const fetchRecommendation = useCallback(async () => {
    try {
      setLoading(true);
      setLoadingState('analyzing');
      setError(null);

      const response = await withTimeout(
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.AGENT_RECOMMEND, { urgency }), {
          method: 'GET'
        }),
        API_CONFIG.TIMEOUT,
        'Request timed out: recommendation'
      );

      if (!response.ok) {
        setLoadingState('error');
        setError('Server error — using live gas data');
        setRecommendation(null);
        return;
      }

      const data = await response.json();

      if (data.success) {
        setRecommendation(data.recommendation);
        setLoadingState('idle');
        setError(null);
        setRetryCount(0);

        if (data.recommendation.action === 'WAIT') {
          const waitMinutes = Math.round((1 - data.recommendation.confidence) * 60);
          setCountdown(waitMinutes * 60);
        } else {
          setCountdown(null);
        }
      } else {
        setLoadingState('error');
        setError(data.error || 'Could not get recommendation');
      }
    } catch (err) {
      console.error('Failed to fetch recommendation:', err);
      const isTimeout = err instanceof Error && (
        err.name === 'TimeoutError' ||
        err.name === 'AbortError' ||
        err.message.includes('timed out')
      );
      setLoadingState(isTimeout ? 'timeout' : 'error');
      setError(isTimeout
        ? 'Analysis taking longer than usual...'
        : 'Connection issue — showing live gas'
      );
      setRetryCount((prev) => prev + 1);
    } finally {
      setLoading(false);
    }
  }, [urgency]);

  // Fetch on mount and set up polling
  useEffect(() => {
    void fetchRecommendation();
    const refreshIfVisible = () => {
      if (typeof document === 'undefined' || document.visibilityState === 'visible') {
        void fetchRecommendation();
      }
    };
    const interval = setInterval(refreshIfVisible, REFRESH_INTERVALS.GAS_DATA);
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        void fetchRecommendation();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [fetchRecommendation]);

  // Countdown timer
  useEffect(() => {
    if (countdown === null || countdown <= 0) return;
    const timer = setInterval(() => {
      setCountdown(prev => (prev !== null && prev > 0 ? prev - 1 : null));
    }, 1000);
    return () => clearInterval(timer);
  }, [countdown]);

  return {
    recommendation,
    loading,
    loadingState,
    error,
    retryCount,
    countdown,
    refresh: fetchRecommendation
  };
};
