/**
 * Custom hook for API health checking.
 *
 * Monitors the backend API health status with automatic polling.
 * Uses React Query for caching and automatic refetching.
 *
 * @module hooks/useApiHealth
 */

import { useQuery } from '@tanstack/react-query';
import { checkHealth } from '../api/gasApi';
import { REFRESH_INTERVALS } from '../constants';

/**
 * Hook to check and monitor API health status.
 *
 * Automatically polls the backend health endpoint at regular intervals
 * to ensure the API is responsive. Useful for showing connection status
 * indicators and gracefully degrading functionality when the API is down.
 *
 * @returns {Object} Query result object from React Query
 * @returns {boolean} returns.data - True if API is healthy, false otherwise
 * @returns {boolean} returns.isLoading - True while the initial health check is in progress
 * @returns {boolean} returns.isError - True if the health check request failed
 * @returns {Error|null} returns.error - Error object if the request failed
 * @returns {Function} returns.refetch - Function to manually trigger a health check
 *
 * @example
 * ```tsx
 * function ConnectionStatus() {
 *   const { data: isHealthy, isLoading } = useApiHealth();
 *
 *   if (isLoading) return <span>Checking connection...</span>;
 *   return <span>{isHealthy ? '🟢 Connected' : '🔴 Disconnected'}</span>;
 * }
 * ```
 */
export function useApiHealth() {
  return useQuery({
    queryKey: ['apiHealth'],
    queryFn: checkHealth,
    refetchInterval: REFRESH_INTERVALS.API_HEALTH,
    staleTime: REFRESH_INTERVALS.API_HEALTH / 2,
  });
}
