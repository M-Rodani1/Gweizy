/**
 * API Error Messages and Utilities
 *
 * Provides user-friendly error messages for API failures.
 * Maps HTTP status codes and error types to actionable messages.
 *
 * @module utils/apiErrors
 */

import { ErrorCode, AppError, APIError } from '../types/errors';

/**
 * Context-specific error messages for different API endpoints.
 */
export const API_ERROR_MESSAGES = {
  // Gas data errors
  gas: {
    fetch: 'Unable to fetch current gas prices. Please try again.',
    stale: 'Gas price data may be outdated. Refresh to get the latest prices.',
    unavailable: 'Gas price service is temporarily unavailable.',
  },

  // Prediction errors
  predictions: {
    fetch: 'Unable to load gas predictions. Try again in a moment.',
    noData: 'No prediction data available for this time range.',
    expired: 'Predictions have expired. Fetching fresh data...',
  },

  // Historical data errors
  historical: {
    fetch: 'Unable to load historical data.',
    range: 'The selected date range is not available.',
    tooLarge: 'Too much data requested. Try a smaller time range.',
  },

  // Transaction errors
  transaction: {
    fetch: 'Unable to load transaction history.',
    submit: 'Failed to submit transaction. Please try again.',
    notFound: 'Transaction not found.',
  },

  // User/Auth errors
  user: {
    unauthorized: 'Please connect your wallet to access this feature.',
    forbidden: 'You don\'t have permission to access this resource.',
    notFound: 'User profile not found.',
  },

  // Network errors
  network: {
    offline: 'You appear to be offline. Some features may be limited.',
    timeout: 'Request timed out. Please check your connection.',
    failed: 'Network request failed. Please try again.',
  },

  // Server errors
  server: {
    internal: 'Our servers are experiencing issues. Please try again later.',
    maintenance: 'Service is under maintenance. Please check back soon.',
    overloaded: 'Service is temporarily overloaded. Please try again in a moment.',
  },

  // Rate limiting
  rateLimit: {
    exceeded: 'Too many requests. Please wait a moment before trying again.',
    quotaReached: 'API quota exceeded. Try again later.',
  },

  // Generic
  generic: {
    unknown: 'An unexpected error occurred. Please try again.',
    retrying: 'Request failed. Retrying...',
    cached: 'Unable to fetch fresh data. Showing cached results.',
  },
} as const;

/**
 * Get user-friendly message for an HTTP status code.
 */
export function getStatusMessage(status: number): string {
  const statusMessages: Record<number, string> = {
    400: 'Invalid request. Please check your input.',
    401: 'Please sign in to continue.',
    403: 'You don\'t have permission to access this.',
    404: 'The requested resource was not found.',
    408: 'Request timed out. Please try again.',
    429: API_ERROR_MESSAGES.rateLimit.exceeded,
    500: API_ERROR_MESSAGES.server.internal,
    502: 'Server is temporarily unreachable.',
    503: API_ERROR_MESSAGES.server.maintenance,
    504: 'Gateway timeout. Please try again.',
  };

  if (status >= 500) {
    return statusMessages[status] || API_ERROR_MESSAGES.server.internal;
  }

  return statusMessages[status] || API_ERROR_MESSAGES.generic.unknown;
}

/**
 * Get user-friendly message for an error code.
 */
export function getErrorCodeMessage(code: ErrorCode): string {
  const codeMessages: Record<ErrorCode, string> = {
    [ErrorCode.NETWORK_ERROR]: API_ERROR_MESSAGES.network.failed,
    [ErrorCode.TIMEOUT]: API_ERROR_MESSAGES.network.timeout,
    [ErrorCode.OFFLINE]: API_ERROR_MESSAGES.network.offline,
    [ErrorCode.API_ERROR]: API_ERROR_MESSAGES.generic.unknown,
    [ErrorCode.RATE_LIMITED]: API_ERROR_MESSAGES.rateLimit.exceeded,
    [ErrorCode.UNAUTHORIZED]: API_ERROR_MESSAGES.user.unauthorized,
    [ErrorCode.FORBIDDEN]: API_ERROR_MESSAGES.user.forbidden,
    [ErrorCode.NOT_FOUND]: 'The requested resource was not found.',
    [ErrorCode.SERVER_ERROR]: API_ERROR_MESSAGES.server.internal,
    [ErrorCode.VALIDATION_ERROR]: 'Please check your input and try again.',
    [ErrorCode.INVALID_INPUT]: 'Invalid input provided.',
    [ErrorCode.INVALID_ADDRESS]: 'The wallet address format is invalid.',
    [ErrorCode.INVALID_CHAIN]: 'The selected blockchain is not supported.',
    [ErrorCode.WEBSOCKET_ERROR]: 'Real-time connection error. Updates may be delayed.',
    [ErrorCode.CONNECTION_FAILED]: 'Failed to establish connection.',
    [ErrorCode.CONNECTION_LOST]: 'Connection lost. Reconnecting...',
    [ErrorCode.PARSE_ERROR]: 'Failed to process server response.',
    [ErrorCode.STALE_DATA]: 'Data may be outdated.',
    [ErrorCode.CACHE_MISS]: 'No cached data available.',
    [ErrorCode.UNKNOWN]: API_ERROR_MESSAGES.generic.unknown,
  };

  return codeMessages[code] || API_ERROR_MESSAGES.generic.unknown;
}

/**
 * API endpoint types for context-specific messages.
 */
export type ApiEndpoint = 'gas' | 'predictions' | 'historical' | 'transaction' | 'user' | 'health' | 'leaderboard' | 'stats';

/**
 * Get a user-friendly error message for an API failure.
 */
export function getApiErrorMessage(
  error: unknown,
  endpoint?: ApiEndpoint
): string {
  // Handle AppError instances
  if (error instanceof AppError) {
    // Use status code message for APIError
    if (error instanceof APIError && error.statusCode) {
      return getStatusMessage(error.statusCode);
    }
    // Use error code message
    return getErrorCodeMessage(error.code);
  }

  // Handle standard Error
  if (error instanceof Error) {
    // Check for network/fetch errors
    if (error.message.includes('fetch') || error.message.includes('network')) {
      return navigator.onLine
        ? API_ERROR_MESSAGES.network.failed
        : API_ERROR_MESSAGES.network.offline;
    }

    // Check for timeout
    if (error.message.includes('timeout') || error.message.includes('AbortError')) {
      return API_ERROR_MESSAGES.network.timeout;
    }

    // Use error message if it's user-friendly
    if (error.message.length < 100 && !error.message.includes('Error:')) {
      return error.message;
    }
  }

  // Use endpoint-specific generic message
  if (endpoint) {
    const endpointMessages: Record<ApiEndpoint, string> = {
      gas: API_ERROR_MESSAGES.gas.fetch,
      predictions: API_ERROR_MESSAGES.predictions.fetch,
      historical: API_ERROR_MESSAGES.historical.fetch,
      transaction: API_ERROR_MESSAGES.transaction.fetch,
      user: API_ERROR_MESSAGES.generic.unknown,
      health: API_ERROR_MESSAGES.server.internal,
      leaderboard: 'Unable to load leaderboard.',
      stats: 'Unable to load statistics.',
    };
    return endpointMessages[endpoint];
  }

  return API_ERROR_MESSAGES.generic.unknown;
}

/**
 * Get a retry message based on error type.
 */
export function getRetryMessage(error: unknown): string | null {
  if (error instanceof AppError && error.isRetryable) {
    return 'This may be a temporary issue. Try again?';
  }

  if (error instanceof APIError) {
    const { statusCode } = error;
    if (statusCode && statusCode >= 500) {
      return 'Server error. Retry might help.';
    }
    if (statusCode === 429) {
      return 'Rate limited. Wait a moment and retry.';
    }
  }

  return null;
}

/**
 * Check if error should show a toast notification.
 */
export function shouldShowErrorToast(error: unknown): boolean {
  // Don't show toast for network offline (we have an offline indicator)
  if (error instanceof AppError && error.code === ErrorCode.OFFLINE) {
    return false;
  }

  // Don't show toast for 401/403 (should redirect to login)
  if (error instanceof APIError) {
    const { statusCode } = error;
    if (statusCode === 401 || statusCode === 403) {
      return false;
    }
  }

  // Don't show toast for stale data (just show indicator)
  if (error instanceof AppError && error.code === ErrorCode.STALE_DATA) {
    return false;
  }

  return true;
}

/**
 * Format error for logging (includes technical details).
 */
export function formatErrorForLogging(error: unknown): string {
  if (error instanceof AppError) {
    return JSON.stringify(error.toJSON(), null, 2);
  }

  if (error instanceof Error) {
    return JSON.stringify({
      name: error.name,
      message: error.message,
      stack: error.stack,
    }, null, 2);
  }

  return String(error);
}

/**
 * Extract error details for Sentry/analytics.
 */
export function getErrorContext(error: unknown): Record<string, unknown> {
  if (error instanceof AppError) {
    return {
      code: error.code,
      isRetryable: error.isRetryable,
      statusCode: error.statusCode,
      details: error.details,
    };
  }

  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
    };
  }

  return { error: String(error) };
}

export default {
  API_ERROR_MESSAGES,
  getStatusMessage,
  getErrorCodeMessage,
  getApiErrorMessage,
  getRetryMessage,
  shouldShowErrorToast,
  formatErrorForLogging,
  getErrorContext,
};
