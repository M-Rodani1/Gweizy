/**
 * Section-level Error Boundary
 * Wraps individual sections to prevent one error from breaking the entire page
 */

import React, { ReactNode } from 'react';
import ErrorBoundary from './ErrorBoundary';

interface SectionErrorBoundaryProps {
  children: ReactNode;
  sectionName: string;
  fallback?: ReactNode;
}

export const SectionErrorBoundary: React.FC<SectionErrorBoundaryProps> = ({
  children,
  sectionName,
  fallback
}) => {
  const defaultFallback = (
    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
      <p className="text-gray-400 text-sm">
        Unable to load {sectionName}. Please refresh the page.
      </p>
    </div>
  );

  return (
    <ErrorBoundary
      fallback={fallback || defaultFallback}
      onError={(error) => {
        console.error(`Error in ${sectionName}:`, error);
      }}
    >
      {children}
    </ErrorBoundary>
  );
};
