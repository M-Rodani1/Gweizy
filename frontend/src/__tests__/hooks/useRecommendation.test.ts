/**
 * Tests for useRecommendation hook
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useRecommendation } from '../../hooks/useRecommendation';

// Mock the API config
vi.mock('../../config/api', () => ({
  API_CONFIG: {
    ENDPOINTS: {
      AGENT_RECOMMEND: '/api/agent/recommend'
    },
    TIMEOUT: 5000
  },
  getApiUrl: (endpoint: string, params: Record<string, unknown>) =>
    `${endpoint}?urgency=${params.urgency}`
}));

describe('useRecommendation', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const mockSuccessResponse = {
    success: true,
    recommendation: {
      action: 'SUBMIT_NOW',
      confidence: 0.85,
      recommended_gas: 25,
      expected_savings: 0.15,
      reasoning: 'Gas prices are low',
      urgency_factor: 0.5
    }
  };

  const mockWaitResponse = {
    success: true,
    recommendation: {
      action: 'WAIT',
      confidence: 0.7,
      recommended_gas: 20,
      expected_savings: 0.25,
      reasoning: 'Prices expected to drop',
      urgency_factor: 0.3
    }
  };

  it('should start with loading state', () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockImplementation(() => new Promise(() => {}));

    const { result } = renderHook(() => useRecommendation(0.5));

    expect(result.current.loading).toBe(true);
    expect(result.current.loadingState).toBe('analyzing');
    expect(result.current.recommendation).toBeNull();
  });

  it('should fetch recommendation on mount', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockSuccessResponse)
    });

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.recommendation).toEqual(mockSuccessResponse.recommendation);
    expect(result.current.loadingState).toBe('idle');
    expect(result.current.error).toBeNull();
  });

  it('should handle server error response', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: false,
      status: 500
    });

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.recommendation).toBeNull();
    expect(result.current.loadingState).toBe('error');
    expect(result.current.error).toBe('Server error — using live gas data');
  });

  it('should handle API error response', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ success: false, error: 'Model unavailable' })
    });

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.loadingState).toBe('error');
    expect(result.current.error).toBe('Model unavailable');
  });

  it('should handle network error and increment retry count', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.loadingState).toBe('error');
    expect(result.current.error).toBe('Connection issue — showing live gas');
    expect(result.current.retryCount).toBe(1);
  });

  it('should handle timeout error', async () => {
    const timeoutError = new Error('Timeout');
    timeoutError.name = 'TimeoutError';
    (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(timeoutError);

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.loadingState).toBe('timeout');
    expect(result.current.error).toBe('Analysis taking longer than usual...');
  });

  it('should set countdown for WAIT action', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockWaitResponse)
    });

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.recommendation?.action).toBe('WAIT');
    // countdown = Math.round((1 - 0.7) * 60) * 60 = 18 * 60 = 1080
    expect(result.current.countdown).toBe(1080);
  });

  it('should not set countdown for non-WAIT action', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockSuccessResponse)
    });

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.recommendation?.action).toBe('SUBMIT_NOW');
    expect(result.current.countdown).toBeNull();
  });

  it('should pass urgency parameter to API', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockSuccessResponse)
    });

    renderHook(() => useRecommendation(0.75));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/agent/recommend?urgency=0.75',
      expect.any(Object)
    );
  });

  it('should provide a refresh function', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockSuccessResponse)
    });

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(typeof result.current.refresh).toBe('function');
  });

  it('should handle default error message when API returns no error', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ success: false })
    });

    const { result } = renderHook(() => useRecommendation(0.5));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe('Could not get recommendation');
  });
});
