/**
 * Custom hook for gas data fetching
 * Uses React Query for caching and automatic refetching
 */

import { useQuery } from '@tanstack/react-query';
import { fetchLiveBaseGas } from '../utils/baseRpc';
import { fetchPredictions } from '../api/gasApi';
import { REFRESH_INTERVALS } from '../constants';

/**
 * Hook to fetch current gas price
 */
export function useCurrentGas() {
  return useQuery({
    queryKey: ['currentGas'],
    queryFn: async () => {
      const data = await fetchLiveBaseGas();
      return data.gwei;
    },
    refetchInterval: REFRESH_INTERVALS.GAS_DATA,
    staleTime: REFRESH_INTERVALS.GAS_DATA / 2,
  });
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
