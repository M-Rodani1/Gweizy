/**
 * Custom hook for gas data fetching
 * Uses WebSocket for real-time updates with React Query fallback
 */

import { useQuery } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { fetchLiveBaseGas } from '../utils/baseRpc';
import { fetchPredictions } from '../api/gasApi';
import { REFRESH_INTERVALS } from '../constants';
import { useWebSocket } from './useWebSocket';

/**
 * Hook to fetch current gas price with WebSocket support
 */
export function useCurrentGas() {
  const { isConnected, gasPrice } = useWebSocket({ enabled: true });
  const [currentGas, setCurrentGas] = useState<number | null>(null);

  // Update from WebSocket
  useEffect(() => {
    if (gasPrice) {
      setCurrentGas(gasPrice.current_gas);
    }
  }, [gasPrice]);

  // Fallback to polling if WebSocket not available
  const fallbackQuery = useQuery({
    queryKey: ['currentGas'],
    queryFn: async () => {
      const data = await fetchLiveBaseGas();
      return data.gwei;
    },
    refetchInterval: isConnected ? false : REFRESH_INTERVALS.GAS_DATA, // Disable polling if WebSocket connected
    staleTime: REFRESH_INTERVALS.GAS_DATA / 2,
    enabled: !isConnected, // Only use polling if WebSocket not connected
  });

  // Use WebSocket data if available, otherwise use fallback
  const finalGas = currentGas ?? fallbackQuery.data ?? 0;

  return {
    data: finalGas,
    isLoading: !isConnected && fallbackQuery.isLoading,
    isError: fallbackQuery.isError,
    error: fallbackQuery.error,
    refetch: fallbackQuery.refetch,
    isWebSocketConnected: isConnected,
  };
}

/**
 * Hook to fetch gas predictions for a specific chain
 * @param chainId - Chain ID (defaults to Base/8453 if not provided)
 */
export function usePredictions(chainId?: number) {
  return useQuery({
    queryKey: ['predictions', chainId],
    queryFn: async () => {
      const result = await fetchPredictions(chainId);
      const preds: { '1h': number; '4h': number; '24h': number } = {
        '1h': 0,
        '4h': 0,
        '24h': 0
      };

      (['1h', '4h', '24h'] as const).forEach((horizon) => {
        const horizonData = result?.predictions?.[horizon];
        if (Array.isArray(horizonData) && horizonData.length > 0 && horizonData[0]?.predictedGwei) {
          preds[horizon] = horizonData[0].predictedGwei;
        }
      });

      return preds;
    },
    refetchInterval: REFRESH_INTERVALS.PREDICTIONS,
    staleTime: REFRESH_INTERVALS.PREDICTIONS / 2,
  });
}

/**
 * Combined hook for gas data for a specific chain
 * @param chainId - Chain ID (defaults to Base/8453 if not provided)
 */
export function useGasData(chainId?: number) {
  const currentGas = useCurrentGas();
  const predictions = usePredictions(chainId);

  return {
    currentGas: currentGas.data ?? 0,
    predictions: predictions.data ?? { '1h': 0, '4h': 0, '24h': 0 },
    isLoading: currentGas.isLoading || predictions.isLoading,
    isError: currentGas.isError || predictions.isError,
    error: currentGas.error || predictions.error,
    refetch: () => {
      currentGas.refetch();
      predictions.refetch();
    }
  };
}
