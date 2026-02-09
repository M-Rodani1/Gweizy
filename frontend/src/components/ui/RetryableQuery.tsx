/**
 * RetryableQuery - A wrapper component for React Query that provides
 * a user-friendly retry UI when queries fail.
 *
 * @module components/ui/RetryableQuery
 */

import React, { ReactNode } from 'react';
import { UseQueryResult } from '@tanstack/react-query';
import { RefreshCw, AlertTriangle, WifiOff } from 'lucide-react';
import { Skeleton } from './Skeleton';
import { isRetryableError, getErrorMessage } from '../../types/errors';

interface RetryableQueryProps<TData> {
  /** The React Query result object */
  query: UseQueryResult<TData, Error>;
  /** Children to render when data is available */
  children: (data: TData) => ReactNode;
  /** Loading state UI */
  loadingFallback?: ReactNode;
  /** Custom error message */
  errorMessage?: string;
  /** Whether to show error details in dev mode */
  showErrorDetails?: boolean;
  /** Optional className for the container */
  className?: string;
  /** Minimum height for loading skeleton */
  minHeight?: string;
}

/**
 * Wrapper component that handles loading, error, and success states
 * for React Query with a built-in retry UI.
 *
 * @example
 * ```tsx
 * <RetryableQuery query={gasQuery}>
 *   {(data) => <GasDisplay data={data} />}
 * </RetryableQuery>
 * ```
 */
export function RetryableQuery<TData>({
  query,
  children,
  loadingFallback,
  errorMessage,
  showErrorDetails = import.meta.env.DEV,
  className = '',
  minHeight = '100px',
}: RetryableQueryProps<TData>) {
  const { data, isLoading, isError, error, refetch, isFetching } = query;

  // Loading state
  if (isLoading) {
    return (
      <div className={className} style={{ minHeight }}>
        {loadingFallback || (
          <div className="flex items-center justify-center h-full">
            <Skeleton className="w-full h-20" />
          </div>
        )}
      </div>
    );
  }

  // Error state
  if (isError) {
    const isOffline = !navigator.onLine;
    const isRetryable = isRetryableError(error);
    const message = errorMessage || getErrorMessage(error);

    return (
      <div
        className={`bg-gray-800/50 border border-red-500/30 rounded-xl p-6 ${className}`}
        role="alert"
        aria-live="polite"
      >
        <div className="flex flex-col items-center text-center gap-4">
          <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
            {isOffline ? (
              <WifiOff className="w-6 h-6 text-red-400" aria-hidden="true" />
            ) : (
              <AlertTriangle className="w-6 h-6 text-red-400" aria-hidden="true" />
            )}
          </div>

          <div>
            <h3 className="text-lg font-medium text-gray-200">
              {isOffline ? 'You appear to be offline' : 'Failed to load data'}
            </h3>
            <p className="text-sm text-gray-400 mt-1">{message}</p>
          </div>

          <button
            onClick={() => refetch()}
            disabled={isFetching}
            type="button"
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-gray-800"
          >
            <RefreshCw
              className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`}
              aria-hidden="true"
            />
            {isFetching ? 'Retrying...' : 'Retry'}
          </button>

          {isRetryable && (
            <p className="text-xs text-gray-500">
              This error may be temporary. Retrying might help.
            </p>
          )}

          {showErrorDetails && error && (
            <details className="w-full mt-2 text-left">
              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                Error details
              </summary>
              <pre className="mt-2 p-3 bg-gray-900/50 rounded text-xs text-red-400 overflow-auto max-h-32">
                {error.toString()}
                {'\n'}
                {error.stack}
              </pre>
            </details>
          )}
        </div>
      </div>
    );
  }

  // Success state - render children with data
  if (data !== undefined) {
    return <>{children(data)}</>;
  }

  // No data (shouldn't normally happen)
  return null;
}

/**
 * Compact retry button for inline error states.
 */
interface RetryButtonProps {
  onRetry: () => void;
  isRetrying?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export function RetryButton({
  onRetry,
  isRetrying = false,
  size = 'md',
  className = '',
}: RetryButtonProps) {
  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1.5 text-sm',
    lg: 'px-4 py-2 text-base',
  };

  const iconSizes = {
    sm: 'w-3 h-3',
    md: 'w-4 h-4',
    lg: 'w-5 h-5',
  };

  return (
    <button
      onClick={onRetry}
      disabled={isRetrying}
      type="button"
      className={`inline-flex items-center gap-1.5 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-gray-200 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-gray-800 ${sizeClasses[size]} ${className}`}
    >
      <RefreshCw
        className={`${iconSizes[size]} ${isRetrying ? 'animate-spin' : ''}`}
        aria-hidden="true"
      />
      {isRetrying ? 'Retrying...' : 'Retry'}
    </button>
  );
}

/**
 * Error message with retry for failed mutations.
 */
interface MutationErrorProps {
  error: Error | null;
  onRetry: () => void;
  isRetrying?: boolean;
  className?: string;
}

export function MutationError({
  error,
  onRetry,
  isRetrying = false,
  className = '',
}: MutationErrorProps) {
  if (!error) return null;

  return (
    <div
      className={`flex items-center justify-between gap-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg ${className}`}
      role="alert"
    >
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" aria-hidden="true" />
        <span className="text-sm text-red-300">{getErrorMessage(error)}</span>
      </div>
      <RetryButton onRetry={onRetry} isRetrying={isRetrying} size="sm" />
    </div>
  );
}

/**
 * Inline error indicator for small spaces.
 */
interface InlineErrorProps {
  message?: string;
  onRetry?: () => void;
}

export function InlineError({ message = 'Failed to load', onRetry }: InlineErrorProps) {
  return (
    <span className="inline-flex items-center gap-1 text-sm text-red-400">
      <AlertTriangle className="w-3.5 h-3.5" aria-hidden="true" />
      {message}
      {onRetry && (
        <button
          onClick={onRetry}
          type="button"
          className="ml-1 text-cyan-400 hover:text-cyan-300 underline focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 rounded"
        >
          Retry
        </button>
      )}
    </span>
  );
}

export default RetryableQuery;
