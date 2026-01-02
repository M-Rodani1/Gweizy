/**
 * Optimistic updates utility
 * Updates UI immediately, rolls back on error
 */

import { featureFlags } from './featureFlags';

interface OptimisticUpdateOptions<T> {
  currentValue: T;
  optimisticValue: T;
  updateFn: () => Promise<T>;
  onSuccess?: (value: T) => void;
  onError?: (error: Error) => void;
  rollbackOnError?: boolean;
}

/**
 * Perform optimistic update
 */
export async function optimisticUpdate<T>(options: OptimisticUpdateOptions<T>): Promise<T> {
  if (!featureFlags.isEnabled('OPTIMISTIC_UPDATES')) {
    // If optimistic updates disabled, just call updateFn
    return options.updateFn();
  }

  const {
    currentValue,
    optimisticValue,
    updateFn,
    onSuccess,
    onError,
    rollbackOnError = true
  } = options;

  // Return optimistic value immediately
  if (onSuccess) {
    onSuccess(optimisticValue);
  }

  try {
    // Perform actual update
    const result = await updateFn();
    
    if (onSuccess) {
      onSuccess(result);
    }
    
    return result;
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    
    // Rollback to current value
    if (rollbackOnError && onSuccess) {
      onSuccess(currentValue);
    }
    
    if (onError) {
      onError(err);
    }
    
    throw err;
  }
}
