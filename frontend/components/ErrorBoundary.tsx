import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    window.location.href = '/';
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-gray-800 border border-red-500/30 rounded-lg p-8 text-center">
            <div className="text-6xl mb-4">😔</div>
            <h1 className="text-2xl font-bold text-white mb-4">Oops! Something went wrong</h1>
            <p className="text-gray-400 mb-6">
              We encountered an unexpected error. Don't worry, your data is safe.
            </p>

            <div className="space-y-3">
              <button
                onClick={this.handleReset}
                className="w-full px-6 py-3 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition"
              >
                Return to Home
              </button>
              
              <button
                onClick={() => window.location.reload()}
                className="w-full px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
              >
                Refresh Page
              </button>
              
              <a
                href="mailto:support@basegasoptimizer.com"
                className="block w-full px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
              >
                Contact Support
              </a>
            </div>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mt-6 text-left">
                <summary className="text-sm text-gray-400 cursor-pointer hover:text-white">
                  Error Details (Dev Mode)
                </summary>
                <pre className="mt-2 p-4 bg-gray-900 rounded text-xs text-red-400 overflow-auto">
                  {this.state.error.toString()}
                  {this.state.errorInfo && this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}

            <p className="mt-6 text-xs text-gray-500">
              Error Code: ERR-{Date.now().toString(36).toUpperCase()}
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

