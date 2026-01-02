/**
 * Custom hook for API health checking
 */

import { useQuery } from '@tanstack/react-query';
import { checkHealth } from '../api/gasApi';
import { REFRESH_INTERVALS } from '../constants';

export function useApiHealth() {
  return useQuery({
    queryKey: ['apiHealth'],
    queryFn: checkHealth,
    refetchInterval: REFRESH_INTERVALS.API_HEALTH,
    staleTime: REFRESH_INTERVALS.API_HEALTH / 2,
  });
}
