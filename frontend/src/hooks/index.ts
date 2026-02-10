/**
 * Hooks barrel exports
 *
 * Re-exports all custom hooks for easier imports:
 * import { usePolling, useDebounce } from '@/hooks';
 *
 * @module hooks
 */

// Data fetching
export { usePolling } from './usePolling';
export { useGasData } from './useGasData';
export { useGasWebSocket, useWebSocket } from './useGasWebSocket';
export { useEthPrice } from './useEthPrice';
export { useApiHealth } from './useApiHealth';

// Form & Input
export { useFormValidation } from './useFormValidation';
export { useDebounce, useDebouncedCallback, useDebouncedSearch } from './useDebounce';
export { useWalletAddress } from './useWalletAddress';

// UI & Performance
export { useLazyLoad } from './useLazyLoad';
export {
  useIntersectionObserver,
  useMultipleIntersectionObserver,
  useInViewport,
} from './useIntersectionObserver';
export { useChartKeyboardNav } from './useChartKeyboardNav';

// Recommendations
export { useRecommendation } from './useRecommendation';

// Transaction scheduling
export { useTransactionScheduler } from './useTransactionScheduler';

// Re-export types
export type { UsePollingOptions, UsePollingResult } from './usePolling';
export type {
  UseGasWebSocketOptions,
  UseGasWebSocketReturn,
  GasPriceUpdate,
  PredictionUpdate,
  MempoolUpdate,
} from './useGasWebSocket';
export type { UseIntersectionObserverOptions } from './useIntersectionObserver';
export type { DebouncedCallbackOptions } from './useDebounce';
export type {
  ScheduleFormState,
  ScheduleFormErrors,
  UseTransactionSchedulerReturn,
} from './useTransactionScheduler';
export type {
  ChartDataPoint,
  UseChartKeyboardNavOptions,
  UseChartKeyboardNavReturn,
} from './useChartKeyboardNav';
