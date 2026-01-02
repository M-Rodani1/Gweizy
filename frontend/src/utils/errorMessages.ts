/**
 * User-friendly error message mapping
 * Converts technical errors into actionable user messages
 */

export interface ErrorInfo {
  message: string;
  action?: string;
  severity: 'error' | 'warning' | 'info';
}

/**
 * Map error to user-friendly message
 */
export function getErrorMessage(error: unknown): ErrorInfo {
  if (error instanceof Error) {
    const errorMessage = error.message.toLowerCase();
    
    // Network errors
    if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
      return {
        message: 'Unable to connect to the network',
        action: 'Please check your internet connection and try again',
        severity: 'error'
      };
    }
    
    // Timeout errors
    if (errorMessage.includes('timeout')) {
      return {
        message: 'Request took too long',
        action: 'The server may be slow. Please try again in a moment',
        severity: 'warning'
      };
    }
    
    // Rate limit errors
    if (errorMessage.includes('rate limit') || errorMessage.includes('429')) {
      return {
        message: 'Too many requests',
        action: 'Please wait a moment before trying again',
        severity: 'warning'
      };
    }
    
    // Server errors
    if (errorMessage.includes('500') || errorMessage.includes('server error')) {
      return {
        message: 'Server error occurred',
        action: 'Our servers are experiencing issues. Please try again later',
        severity: 'error'
      };
    }
    
    // Not found errors
    if (errorMessage.includes('404') || errorMessage.includes('not found')) {
      return {
        message: 'Resource not found',
        action: 'The requested data could not be found',
        severity: 'warning'
      };
    }
    
    // RPC errors
    if (errorMessage.includes('rpc') || errorMessage.includes('invalid rpc')) {
      return {
        message: 'Blockchain connection issue',
        action: 'Unable to fetch live gas prices. Using cached data',
        severity: 'warning'
      };
    }
  }
  
  // Default error
  return {
    message: 'Something went wrong',
    action: 'Please try again or refresh the page',
    severity: 'error'
  };
}

/**
 * Check if error is retryable
 */
export function isRetryableError(error: unknown): boolean {
  if (error instanceof Error) {
    const message = error.message.toLowerCase();
    return (
      message.includes('network') ||
      message.includes('timeout') ||
      message.includes('500') ||
      message.includes('503')
    );
  }
  return false;
}
