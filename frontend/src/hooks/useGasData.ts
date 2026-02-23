/**
 * Custom hooks for gas data fetching.
 *
 * Provides hooks for fetching current gas prices and predictions,
 * with real-time WebSocket support and React Query polling fallback.
 *
 * @module hooks/useGasData
 */

import { useQuery } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { fetchLiveBaseGas } from '../utils/baseRpc';
import { fetchPredictions } from '../api/gasApi';
import { FEATURE_FLAGS, REFRESH_INTERVALS } from '../constants';
import { useGasWebSocket } from './useGasWebSocket';

/**
 * Hook to fetch current gas price with WebSocket support.
 *
 * Uses WebSocket for real-time updates when available, with automatic
 * fallback to polling via React Query when WebSocket is not connected.
 *
 * @returns {Object} Gas price state and controls
 * @returns {number} returns.data - Current gas price in gwei
 * @returns {boolean} returns.isLoading - True while fetching initial data
 * @returns {boolean} returns.isError - True if fetch failed
 * @returns {Error|null} returns.error - Error object if fetch failed
 * @returns {Function} returns.refetch - Function to manually refresh
 * @returns {boolean} returns.isWebSocketConnected - WebSocket connection status
 *
 * @example
 * ```tsx
 * function GasPrice() {
 *   const { data: gasPrice, isLoading, isWebSocketConnected } = useCurrentGas();
 *
 *   return (
 *     <div>
 *       <span>{gasPrice.toFixed(4)} gwei</span>
 *       {isWebSocketConnected && <span>🔴 Live</span>}
 *     </div>
 *   );
 * }
 * ```
 */
export function useCurrentGas() {
  const { isConnected, gasPrice } = useGasWebSocket({ enabled: FEATURE_FLAGS.WEBSOCKET_ENABLED });
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
 * Hook to fetch gas price predictions for a specific chain.
 *
 * Fetches ML model predictions for gas prices at 1h, 4h, and 24h horizons.
 * Automatically polls at regular intervals and caches results.
 *
 * @param {number} [chainId] - Chain ID (defaults to Base/8453 if not provided)
 *
 * @returns {Object} Query result with prediction data
 * @returns {Object} returns.data - Predictions object with 1h, 4h, 24h keys
 * @returns {boolean} returns.isLoading - True while fetching
 * @returns {boolean} returns.isError - True if fetch failed
 * @returns {Function} returns.refetch - Function to manually refresh
 *
 * @example
 * ```tsx
 * function PredictionDisplay() {
 *   const { data: predictions, isLoading } = usePredictions(8453);
 *
 *   if (isLoading) return <Skeleton />;
 *
 *   return (
 *     <div>
 *       <p>1h: {predictions['1h'].toFixed(4)} gwei</p>
 *       <p>4h: {predictions['4h'].toFixed(4)} gwei</p>
 *       <p>24h: {predictions['24h'].toFixed(4)} gwei</p>
 *     </div>
 *   );
 * }
 * ```
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
 * Combined hook for all gas data (current price + predictions).
 *
 * Convenience hook that combines useCurrentGas and usePredictions
 * into a single interface. Ideal for components that need both
 * current price and predictions.
 *
 * @param {number} [chainId] - Chain ID (defaults to Base/8453 if not provided)
 *
 * @returns {Object} Combined gas data state
 * @returns {number} returns.currentGas - Current gas price in gwei (0 if unavailable)
 * @returns {Object} returns.predictions - Predictions for 1h, 4h, 24h horizons
 * @returns {boolean} returns.isLoading - True while either query is loading
 * @returns {boolean} returns.isError - True if either query failed
 * @returns {Error|null} returns.error - First error encountered
 * @returns {Function} returns.refetch - Function to refresh all data
 *
 * @example
 * ```tsx
 * function GasDashboard() {
 *   const {
 *     currentGas,
 *     predictions,
 *     isLoading,
 *     refetch
 *   } = useGasData(8453);
 *
 *   if (isLoading) return <LoadingSpinner />;
 *
 *   return (
 *     <div>
 *       <CurrentPrice value={currentGas} />
 *       <PredictionCard horizon="1h" value={predictions['1h']} />
 *       <button onClick={refetch}>Refresh</button>
 *     </div>
 *   );
 * }
 * ```
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
