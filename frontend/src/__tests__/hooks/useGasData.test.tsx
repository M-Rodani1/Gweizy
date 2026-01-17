/**
 * Tests for useGasData hooks
 *
 * Tests the gas data fetching hooks including:
 * - useCurrentGas: WebSocket + polling fallback
 * - usePredictions: Prediction data fetching
 * - useGasData: Combined hook
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';

// Mock the dependencies
vi.mock('../../api/gasApi', () => ({
  fetchPredictions: vi.fn(),
}));

vi.mock('../../utils/baseRpc', () => ({
  fetchLiveBaseGas: vi.fn(),
}));

vi.mock('../../hooks/useWebSocket', () => ({
  useWebSocket: vi.fn(() => ({
    isConnected: false,
    gasPrice: null,
  })),
}));

// Import after mocks
import { useCurrentGas, usePredictions, useGasData } from '../../hooks/useGasData';
import { fetchPredictions } from '../../api/gasApi';
import { fetchLiveBaseGas } from '../../utils/baseRpc';
import { useWebSocket } from '../../hooks/useWebSocket';

// Create wrapper with QueryClient
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useGasData hooks', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('useCurrentGas', () => {
    it('should return loading state initially', () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      (fetchLiveBaseGas as any).mockResolvedValue({ gwei: 0.001 });

      const { result } = renderHook(() => useCurrentGas(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
    });

    it('should fetch gas price via polling when WebSocket not connected', async () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      (fetchLiveBaseGas as any).mockResolvedValue({ gwei: 0.00234 });

      const { result } = renderHook(() => useCurrentGas(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data).toBe(0.00234);
      expect(fetchLiveBaseGas).toHaveBeenCalled();
    });

    it('should use WebSocket data when connected', async () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: true,
        gasPrice: { current_gas: 0.00345 },
      });

      const { result } = renderHook(() => useCurrentGas(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.data).toBe(0.00345);
      });

      expect(result.current.isWebSocketConnected).toBe(true);
      // Should not call polling when WebSocket is connected
      expect(fetchLiveBaseGas).not.toHaveBeenCalled();
    });

    it('should handle fetch errors gracefully', async () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      (fetchLiveBaseGas as any).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useCurrentGas(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toBeTruthy();
    });

    it('should provide refetch function', async () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      (fetchLiveBaseGas as any).mockResolvedValue({ gwei: 0.001 });

      const { result } = renderHook(() => useCurrentGas(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(typeof result.current.refetch).toBe('function');
    });
  });

  describe('usePredictions', () => {
    it('should fetch predictions successfully', async () => {
      const mockPredictions = {
        predictions: {
          '1h': [{ predictedGwei: 0.0021 }],
          '4h': [{ predictedGwei: 0.0025 }],
          '24h': [{ predictedGwei: 0.0030 }],
        },
      };
      (fetchPredictions as any).mockResolvedValue(mockPredictions);

      const { result } = renderHook(() => usePredictions(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data).toEqual({
        '1h': 0.0021,
        '4h': 0.0025,
        '24h': 0.0030,
      });
    });

    it('should handle missing prediction data', async () => {
      const mockPredictions = {
        predictions: {},
      };
      (fetchPredictions as any).mockResolvedValue(mockPredictions);

      const { result } = renderHook(() => usePredictions(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.data).toEqual({
        '1h': 0,
        '4h': 0,
        '24h': 0,
      });
    });

    it('should pass chainId to fetchPredictions', async () => {
      const mockPredictions = {
        predictions: {
          '1h': [{ predictedGwei: 0.002 }],
          '4h': [{ predictedGwei: 0.002 }],
          '24h': [{ predictedGwei: 0.002 }],
        },
      };
      (fetchPredictions as any).mockResolvedValue(mockPredictions);

      const chainId = 8453;
      renderHook(() => usePredictions(chainId), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(fetchPredictions).toHaveBeenCalledWith(chainId);
      });
    });

    it('should handle fetch errors', async () => {
      (fetchPredictions as any).mockRejectedValue(new Error('API error'));

      const { result } = renderHook(() => usePredictions(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });
    });
  });

  describe('useGasData', () => {
    it('should combine current gas and predictions', async () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      (fetchLiveBaseGas as any).mockResolvedValue({ gwei: 0.002 });
      (fetchPredictions as any).mockResolvedValue({
        predictions: {
          '1h': [{ predictedGwei: 0.0021 }],
          '4h': [{ predictedGwei: 0.0025 }],
          '24h': [{ predictedGwei: 0.0030 }],
        },
      });

      const { result } = renderHook(() => useGasData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.currentGas).toBe(0.002);
      expect(result.current.predictions).toEqual({
        '1h': 0.0021,
        '4h': 0.0025,
        '24h': 0.0030,
      });
    });

    it('should show loading when either query is loading', () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      // Don't resolve - keep loading
      (fetchLiveBaseGas as any).mockImplementation(() => new Promise(() => {}));
      (fetchPredictions as any).mockResolvedValue({
        predictions: {
          '1h': [{ predictedGwei: 0.002 }],
          '4h': [{ predictedGwei: 0.002 }],
          '24h': [{ predictedGwei: 0.002 }],
        },
      });

      const { result } = renderHook(() => useGasData(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);
    });

    it('should show error if either query has error', async () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      (fetchLiveBaseGas as any).mockRejectedValue(new Error('Error'));
      (fetchPredictions as any).mockResolvedValue({
        predictions: {
          '1h': [{ predictedGwei: 0.002 }],
          '4h': [{ predictedGwei: 0.002 }],
          '24h': [{ predictedGwei: 0.002 }],
        },
      });

      const { result } = renderHook(() => useGasData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });
    });

    it('should provide refetch function that refreshes both queries', async () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      (fetchLiveBaseGas as any).mockResolvedValue({ gwei: 0.002 });
      (fetchPredictions as any).mockResolvedValue({
        predictions: {
          '1h': [{ predictedGwei: 0.002 }],
          '4h': [{ predictedGwei: 0.002 }],
          '24h': [{ predictedGwei: 0.002 }],
        },
      });

      const { result } = renderHook(() => useGasData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(typeof result.current.refetch).toBe('function');

      // Call refetch
      result.current.refetch();

      // Both should be called again
      await waitFor(() => {
        expect(fetchLiveBaseGas).toHaveBeenCalledTimes(2);
      });
    });

    it('should return default values when data is null', async () => {
      (useWebSocket as any).mockReturnValue({
        isConnected: false,
        gasPrice: null,
      });
      (fetchLiveBaseGas as any).mockResolvedValue({ gwei: null });
      (fetchPredictions as any).mockResolvedValue(null);

      const { result } = renderHook(() => useGasData(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Should return defaults
      expect(result.current.currentGas).toBe(0);
      expect(result.current.predictions).toEqual({ '1h': 0, '4h': 0, '24h': 0 });
    });
  });
});
