/**
 * Standardized error types for the application.
 *
 * Provides a consistent error handling pattern across the codebase.
 *
 * @module types/errors
 */

/**
 * Error codes for categorizing errors.
 */
export enum ErrorCode {
  // Network errors
  NETWORK_ERROR = 'NETWORK_ERROR',
  TIMEOUT = 'TIMEOUT',
  OFFLINE = 'OFFLINE',

  // API errors
  API_ERROR = 'API_ERROR',
  RATE_LIMITED = 'RATE_LIMITED',
  UNAUTHORIZED = 'UNAUTHORIZED',
  FORBIDDEN = 'FORBIDDEN',
  NOT_FOUND = 'NOT_FOUND',
  SERVER_ERROR = 'SERVER_ERROR',

  // Validation errors
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  INVALID_INPUT = 'INVALID_INPUT',
  INVALID_ADDRESS = 'INVALID_ADDRESS',
  INVALID_CHAIN = 'INVALID_CHAIN',

  // WebSocket errors
  WEBSOCKET_ERROR = 'WEBSOCKET_ERROR',
  CONNECTION_FAILED = 'CONNECTION_FAILED',
  CONNECTION_LOST = 'CONNECTION_LOST',

  // Data errors
  PARSE_ERROR = 'PARSE_ERROR',
  STALE_DATA = 'STALE_DATA',
  CACHE_MISS = 'CACHE_MISS',

  // Unknown
  UNKNOWN = 'UNKNOWN',
}

/**
 * Base application error class.
 */
export class AppError extends Error {
  readonly code: ErrorCode;
  readonly isRetryable: boolean;
  readonly statusCode?: number;
  readonly details?: Record<string, unknown>;
  readonly timestamp: Date;

  constructor(
    message: string,
    code: ErrorCode = ErrorCode.UNKNOWN,
    options: {
      isRetryable?: boolean;
      statusCode?: number;
      details?: Record<string, unknown>;
      cause?: Error;
    } = {}
  ) {
    super(message);
    this.name = 'AppError';
    this.code = code;
    this.isRetryable = options.isRetryable ?? false;
    this.statusCode = options.statusCode;
    this.details = options.details;
    this.timestamp = new Date();

    if (options.cause) {
      this.cause = options.cause;
    }

    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      isRetryable: this.isRetryable,
      statusCode: this.statusCode,
      details: this.details,
      timestamp: this.timestamp.toISOString(),
    };
  }
}

/**
 * API-specific error class.
 */
export class APIError extends AppError {
  readonly endpoint?: string;
  readonly method?: string;

  constructor(
    message: string,
    statusCode: number,
    options: {
      endpoint?: string;
      method?: string;
      details?: Record<string, unknown>;
      cause?: Error;
    } = {}
  ) {
    const code = statusCodeToErrorCode(statusCode);
    const isRetryable = statusCode >= 500 || statusCode === 429;

    super(message, code, {
      isRetryable,
      statusCode,
      details: options.details,
      cause: options.cause,
    });

    this.name = 'APIError';
    this.endpoint = options.endpoint;
    this.method = options.method;
  }
}

/**
 * Network error class.
 */
export class NetworkError extends AppError {
  constructor(message = 'Network request failed', cause?: Error) {
    super(message, ErrorCode.NETWORK_ERROR, {
      isRetryable: true,
      cause,
    });
    this.name = 'NetworkError';
  }
}

/**
 * Timeout error class.
 */
export class TimeoutError extends AppError {
  readonly timeoutMs: number;

  constructor(message = 'Request timed out', timeoutMs: number, cause?: Error) {
    super(message, ErrorCode.TIMEOUT, {
      isRetryable: true,
      details: { timeoutMs },
      cause,
    });
    this.name = 'TimeoutError';
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Validation error class.
 */
export class ValidationError extends AppError {
  readonly field?: string;
  readonly value?: unknown;

  constructor(
    message: string,
    options: {
      field?: string;
      value?: unknown;
      details?: Record<string, unknown>;
    } = {}
  ) {
    super(message, ErrorCode.VALIDATION_ERROR, {
      isRetryable: false,
      details: { ...options.details, field: options.field },
    });
    this.name = 'ValidationError';
    this.field = options.field;
    this.value = options.value;
  }
}

/**
 * WebSocket error class.
 */
export class WebSocketError extends AppError {
  readonly reconnectAttempts?: number;

  constructor(
    message: string,
    code: ErrorCode = ErrorCode.WEBSOCKET_ERROR,
    options: {
      reconnectAttempts?: number;
      isRetryable?: boolean;
      cause?: Error;
    } = {}
  ) {
    super(message, code, {
      isRetryable: options.isRetryable ?? true,
      cause: options.cause,
    });
    this.name = 'WebSocketError';
    this.reconnectAttempts = options.reconnectAttempts;
  }
}

/**
 * Map HTTP status code to error code.
 */
function statusCodeToErrorCode(statusCode: number): ErrorCode {
  if (statusCode === 401) return ErrorCode.UNAUTHORIZED;
  if (statusCode === 403) return ErrorCode.FORBIDDEN;
  if (statusCode === 404) return ErrorCode.NOT_FOUND;
  if (statusCode === 429) return ErrorCode.RATE_LIMITED;
  if (statusCode >= 500) return ErrorCode.SERVER_ERROR;
  return ErrorCode.API_ERROR;
}

/**
 * Type guard to check if error is an AppError.
 */
export function isAppError(error: unknown): error is AppError {
  return error instanceof AppError;
}

/**
 * Type guard to check if error is an APIError.
 */
export function isAPIError(error: unknown): error is APIError {
  return error instanceof APIError;
}

/**
 * Type guard to check if error is retryable.
 */
export function isRetryableError(error: unknown): boolean {
  if (isAppError(error)) {
    return error.isRetryable;
  }
  // Network errors are generally retryable
  if (error instanceof TypeError && error.message.includes('fetch')) {
    return true;
  }
  return false;
}

/**
 * Convert unknown error to AppError.
 */
export function toAppError(error: unknown): AppError {
  if (isAppError(error)) {
    return error;
  }

  if (error instanceof Error) {
    return new AppError(error.message, ErrorCode.UNKNOWN, { cause: error });
  }

  if (typeof error === 'string') {
    return new AppError(error);
  }

  return new AppError('An unknown error occurred');
}

/**
 * Get user-friendly error message.
 */
export function getErrorMessage(error: unknown): string {
  if (isAppError(error)) {
    return error.message;
  }

  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'An unexpected error occurred. Please try again.';
}

/**
 * Error messages for common error codes.
 */
export const ERROR_MESSAGES: Record<ErrorCode, string> = {
  [ErrorCode.NETWORK_ERROR]: 'Unable to connect to the server. Please check your connection.',
  [ErrorCode.TIMEOUT]: 'The request took too long. Please try again.',
  [ErrorCode.OFFLINE]: 'You appear to be offline. Some features may be limited.',
  [ErrorCode.API_ERROR]: 'An error occurred while fetching data.',
  [ErrorCode.RATE_LIMITED]: 'Too many requests. Please wait a moment and try again.',
  [ErrorCode.UNAUTHORIZED]: 'You need to sign in to access this feature.',
  [ErrorCode.FORBIDDEN]: 'You do not have permission to access this resource.',
  [ErrorCode.NOT_FOUND]: 'The requested resource was not found.',
  [ErrorCode.SERVER_ERROR]: 'The server encountered an error. Please try again later.',
  [ErrorCode.VALIDATION_ERROR]: 'The provided data is invalid.',
  [ErrorCode.INVALID_INPUT]: 'Please check your input and try again.',
  [ErrorCode.INVALID_ADDRESS]: 'The wallet address is invalid.',
  [ErrorCode.INVALID_CHAIN]: 'The selected chain is not supported.',
  [ErrorCode.WEBSOCKET_ERROR]: 'Real-time connection error.',
  [ErrorCode.CONNECTION_FAILED]: 'Failed to establish connection.',
  [ErrorCode.CONNECTION_LOST]: 'Connection lost. Attempting to reconnect...',
  [ErrorCode.PARSE_ERROR]: 'Failed to process the response data.',
  [ErrorCode.STALE_DATA]: 'The data may be outdated.',
  [ErrorCode.CACHE_MISS]: 'No cached data available.',
  [ErrorCode.UNKNOWN]: 'An unexpected error occurred.',
};

export default {
  ErrorCode,
  AppError,
  APIError,
  NetworkError,
  TimeoutError,
  ValidationError,
  WebSocketError,
  isAppError,
  isAPIError,
  isRetryableError,
  toAppError,
  getErrorMessage,
  ERROR_MESSAGES,
};
