/**
 * Optimistic updates utilities for React Query.
 *
 * Provides patterns for updating the UI immediately while syncing with the server.
 * Includes rollback functionality for failed mutations.
 *
 * @module utils/optimisticUpdates
 */

import type { QueryClient } from '@tanstack/react-query';
import { isFeatureEnabled } from './featureFlags';

// ========================================
// Basic Optimistic Update
// ========================================

interface OptimisticUpdateOptions<T> {
  currentValue: T;
  optimisticValue: T;
  updateFn: () => Promise<T>;
  onSuccess?: (value: T) => void;
  onError?: (error: Error) => void;
  rollbackOnError?: boolean;
}

/**
 * Perform optimistic update (basic version).
 */
export async function optimisticUpdate<T>(options: OptimisticUpdateOptions<T>): Promise<T> {
  if (!isFeatureEnabled('OPTIMISTIC_UPDATES')) {
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

  if (onSuccess) {
    onSuccess(optimisticValue);
  }

  try {
    const result = await updateFn();

    if (onSuccess) {
      onSuccess(result);
    }

    return result;
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));

    if (rollbackOnError && onSuccess) {
      onSuccess(currentValue);
    }

    if (onError) {
      onError(err);
    }

    throw err;
  }
}

// ========================================
// React Query Integration
// ========================================

/**
 * Configuration for an optimistic mutation.
 */
export interface OptimisticMutationConfig<TData, TError, TVariables, TContext> {
  queryKey: unknown[];
  optimisticUpdate: (oldData: TData | undefined, variables: TVariables) => TData;
  onSuccess?: (data: TData, variables: TVariables, context: TContext) => void;
  onError?: (error: TError, variables: TVariables, context: TContext) => void;
  onSettled?: (
    data: TData | undefined,
    error: TError | null,
    variables: TVariables,
    context: TContext
  ) => void;
}

/**
 * Create optimistic mutation options for React Query.
 *
 * @example
 * ```tsx
 * const mutation = useMutation({
 *   mutationFn: updatePreferences,
 *   ...createOptimisticMutation(queryClient, {
 *     queryKey: ['preferences'],
 *     optimisticUpdate: (old, newPrefs) => ({ ...old, ...newPrefs }),
 *   }),
 * });
 * ```
 */
export function createOptimisticMutation<TData, TError, TVariables>(
  queryClient: QueryClient,
  config: OptimisticMutationConfig<TData, TError, TVariables, { previousData: TData | undefined }>
) {
  if (!isFeatureEnabled('OPTIMISTIC_UPDATES')) {
    return {
      onSettled: () => {
        queryClient.invalidateQueries({ queryKey: config.queryKey });
      },
    };
  }

  return {
    onMutate: async (variables: TVariables) => {
      await queryClient.cancelQueries({ queryKey: config.queryKey });
      const previousData = queryClient.getQueryData<TData>(config.queryKey);

      queryClient.setQueryData<TData>(config.queryKey, (old) =>
        config.optimisticUpdate(old, variables)
      );

      return { previousData };
    },

    onError: (
      error: TError,
      variables: TVariables,
      context: { previousData: TData | undefined } | undefined
    ) => {
      if (context?.previousData !== undefined) {
        queryClient.setQueryData(config.queryKey, context.previousData);
      }
      config.onError?.(error, variables, context ?? { previousData: undefined });
    },

    onSuccess: (data: TData, variables: TVariables, context: { previousData: TData | undefined }) => {
      config.onSuccess?.(data, variables, context);
    },

    onSettled: (
      data: TData | undefined,
      error: TError | null,
      variables: TVariables,
      context: { previousData: TData | undefined } | undefined
    ) => {
      queryClient.invalidateQueries({ queryKey: config.queryKey });
      config.onSettled?.(data, error, variables, context ?? { previousData: undefined });
    },
  };
}

/**
 * Create optimistic update for adding an item to a list.
 */
export function createOptimisticAdd<TItem extends { id: string }>(
  queryClient: QueryClient,
  queryKey: unknown[]
) {
  return createOptimisticMutation<TItem[], Error, Omit<TItem, 'id'> & { id?: string }>(
    queryClient,
    {
      queryKey,
      optimisticUpdate: (oldData, newItem) => {
        const tempId = newItem.id ?? `temp-${Date.now()}`;
        const itemWithId = { ...newItem, id: tempId } as TItem;
        return [...(oldData ?? []), itemWithId];
      },
    }
  );
}

/**
 * Create optimistic update for updating an item in a list.
 */
export function createOptimisticListUpdate<TItem extends { id: string }>(
  queryClient: QueryClient,
  queryKey: unknown[]
) {
  return createOptimisticMutation<TItem[], Error, Partial<TItem> & { id: string }>(queryClient, {
    queryKey,
    optimisticUpdate: (oldData, updates) => {
      if (!oldData) return [];
      return oldData.map((item) =>
        item.id === updates.id ? { ...item, ...updates } : item
      );
    },
  });
}

/**
 * Create optimistic update for removing an item from a list.
 */
export function createOptimisticRemove<TItem extends { id: string }>(
  queryClient: QueryClient,
  queryKey: unknown[]
) {
  return createOptimisticMutation<TItem[], Error, { id: string }>(queryClient, {
    queryKey,
    optimisticUpdate: (oldData, { id }) => {
      if (!oldData) return [];
      return oldData.filter((item) => item.id !== id);
    },
  });
}

/**
 * Create optimistic update for toggling a boolean field.
 */
export function createOptimisticToggle<TItem extends { id: string }>(
  queryClient: QueryClient,
  queryKey: unknown[],
  field: keyof TItem
) {
  return createOptimisticMutation<TItem[], Error, { id: string }>(queryClient, {
    queryKey,
    optimisticUpdate: (oldData, { id }) => {
      if (!oldData) return [];
      return oldData.map((item) =>
        item.id === id ? { ...item, [field]: !item[field] } : item
      );
    },
  });
}

// ========================================
// Scheduled Transactions Mutations
// ========================================

export interface ScheduledTransaction {
  id: string;
  chainId: number;
  txType: string;
  targetGasPrice: number;
  maxGasPrice?: number;
  status: 'pending' | 'ready' | 'executed' | 'expired' | 'cancelled';
  createdAt: number;
  expiresAt?: number;
}

/**
 * Create optimistic mutations for scheduled transactions.
 */
export function createScheduledTxMutations(queryClient: QueryClient) {
  const queryKey = ['scheduledTransactions'];

  return {
    add: createOptimisticMutation<
      ScheduledTransaction[],
      Error,
      Omit<ScheduledTransaction, 'id' | 'status' | 'createdAt'>
    >(queryClient, {
      queryKey,
      optimisticUpdate: (oldData, newTx) => {
        const tx: ScheduledTransaction = {
          ...newTx,
          id: `temp-${Date.now()}`,
          status: 'pending',
          createdAt: Date.now(),
        };
        return [...(oldData ?? []), tx];
      },
    }),

    update: createOptimisticMutation<
      ScheduledTransaction[],
      Error,
      Partial<ScheduledTransaction> & { id: string }
    >(queryClient, {
      queryKey,
      optimisticUpdate: (oldData, updates) => {
        if (!oldData) return [];
        return oldData.map((tx) =>
          tx.id === updates.id ? { ...tx, ...updates } : tx
        );
      },
    }),

    cancel: createOptimisticMutation<ScheduledTransaction[], Error, { id: string }>(
      queryClient,
      {
        queryKey,
        optimisticUpdate: (oldData, { id }) => {
          if (!oldData) return [];
          return oldData.map((tx) =>
            tx.id === id ? { ...tx, status: 'cancelled' as const } : tx
          );
        },
      }
    ),

    remove: createOptimisticRemove<ScheduledTransaction>(queryClient, queryKey),
  };
}

// ========================================
// Batch Updates
// ========================================

/**
 * Batch optimistic updates for multiple queries.
 */
export function batchOptimisticUpdates(
  queryClient: QueryClient,
  updates: Array<{
    queryKey: unknown[];
    updater: (old: unknown) => unknown;
  }>
): () => void {
  const snapshots = updates.map(({ queryKey }) => ({
    queryKey,
    data: queryClient.getQueryData(queryKey),
  }));

  updates.forEach(({ queryKey, updater }) => {
    queryClient.setQueryData(queryKey, updater);
  });

  return () => {
    snapshots.forEach(({ queryKey, data }) => {
      queryClient.setQueryData(queryKey, data);
    });
  };
}

// ========================================
// Debounced Updates
// ========================================

/**
 * Create a debounced optimistic update for frequent changes.
 */
export function createDebouncedOptimisticUpdate<TData, TVariables>(
  queryClient: QueryClient,
  queryKey: unknown[],
  mutationFn: (variables: TVariables) => Promise<TData>,
  delay = 500
) {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  let latestVariables: TVariables | null = null;
  let originalData: TData | undefined;
  let isFirstUpdate = true;

  const debouncedMutate = async (
    variables: TVariables,
    updater: (old: TData | undefined) => TData
  ): Promise<TData> => {
    if (isFirstUpdate) {
      originalData = queryClient.getQueryData<TData>(queryKey);
      isFirstUpdate = false;
    }

    queryClient.setQueryData<TData>(queryKey, updater);
    latestVariables = variables;

    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    return new Promise<TData>((resolve, reject) => {
      timeoutId = setTimeout(async () => {
        try {
          const result = await mutationFn(latestVariables!);
          isFirstUpdate = true;
          resolve(result);
        } catch (error) {
          if (originalData !== undefined) {
            queryClient.setQueryData(queryKey, originalData);
          }
          isFirstUpdate = true;
          reject(error);
        }
      }, delay);
    });
  };

  const cancel = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    if (originalData !== undefined && !isFirstUpdate) {
      queryClient.setQueryData(queryKey, originalData);
    }
    isFirstUpdate = true;
  };

  return { debouncedMutate, cancel };
}

export default {
  optimisticUpdate,
  createOptimisticMutation,
  createOptimisticAdd,
  createOptimisticListUpdate,
  createOptimisticRemove,
  createOptimisticToggle,
  createScheduledTxMutations,
  batchOptimisticUpdates,
  createDebouncedOptimisticUpdate,
}
