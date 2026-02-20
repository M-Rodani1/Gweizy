/**
 * Section-level Error Boundary
 *
 * Lightweight wrapper for individual sections to prevent one error
 * from breaking the entire page. Shows a compact error UI with retry option.
 */

import { Component, ReactNode, ErrorInfo } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import * as Sentry from '@sentry/react';

interface SectionErrorBoundaryProps {
  children: ReactNode;
  /** Name of the section for error messages */
  sectionName: string;
  /** Optional custom fallback UI */
  fallback?: ReactNode;
  /** Optional callback when error occurs */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Optional className for the error container */
  className?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Section-level error boundary that contains failures within individual
 * components/sections without crashing the entire application.
 *
 * @example
 * <SectionErrorBoundary sectionName="Gas Predictions">
 *   <GasPredictionCard />
 * </SectionErrorBoundary>
 *
 * @example
 * // With custom fallback
 * <SectionErrorBoundary
 *   sectionName="Chart"
 *   fallback={<PlaceholderChart />}
 * >
 *   <ComplexChart data={data} />
 * </SectionErrorBoundary>
 */
export class SectionErrorBoundary extends Component<SectionErrorBoundaryProps, State> {
  public override state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public override componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const { sectionName, onError } = this.props;

    // Log with section context
    console.error(`Error in ${sectionName}:`, error, errorInfo);

    // Report to Sentry with section context
    Sentry.captureException(error, {
      tags: {
        section: sectionName,
      },
      extra: {
        componentStack: errorInfo.componentStack,
      },
    });

    // Call optional error callback
    onError?.(error, errorInfo);
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  public override render() {
    const { children, sectionName, fallback, className = '' } = this.props;

    if (this.state.hasError) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback;
      }

      // Default compact error UI
      return (
        <div
          className={`bg-gray-800/50 border border-red-500/30 rounded-xl p-4 ${className}`}
          role="alert"
          aria-live="polite"
        >
          <div className="flex items-center gap-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-red-400" aria-hidden="true" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-200">
                Unable to load {sectionName}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">
                An error occurred while rendering this section.
              </p>
            </div>
            <button
              onClick={this.handleRetry}
              type="button"
              className="flex-shrink-0 px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-gray-800 flex items-center gap-1.5"
            >
              <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" />
              Retry
            </button>
          </div>

          {import.meta.env.DEV && this.state.error && (
            <details className="mt-3 pt-3 border-t border-gray-700/50">
              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                Error details
              </summary>
              <pre className="mt-2 p-2 bg-gray-900/50 rounded text-xs text-red-400 overflow-auto max-h-32">
                {this.state.error.toString()}
              </pre>
            </details>
          )}
        </div>
      );
    }

    // Wrap children in div if className is provided for consistent styling
    if (className) {
      return <div className={className}>{children}</div>;
    }

    return children;
  }
}

export default SectionErrorBoundary;
