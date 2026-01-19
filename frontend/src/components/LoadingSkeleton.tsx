import React, { memo } from 'react';

// Memoized skeleton components to prevent unnecessary re-renders
export const CardSkeleton = memo(() => (
  <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 animate-pulse">
    <div className="h-6 bg-gray-700 rounded w-3/4 mb-4"></div>
    <div className="h-4 bg-gray-700 rounded w-1/2 mb-2"></div>
    <div className="h-4 bg-gray-700 rounded w-2/3"></div>
  </div>
));
CardSkeleton.displayName = 'CardSkeleton';

export const GasPriceCardSkeleton = memo(() => (
  <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 animate-pulse">
    <div className="flex justify-between items-center mb-4">
      <div className="h-6 bg-gray-700 rounded w-32"></div>
      <div className="h-6 bg-gray-700 rounded w-20"></div>
    </div>
    <div className="text-center mb-4">
      <div className="h-12 bg-gray-700 rounded w-40 mx-auto mb-2"></div>
      <div className="h-4 bg-gray-700 rounded w-24 mx-auto"></div>
    </div>
    <div className="space-y-2">
      <div className="h-3 bg-gray-700 rounded w-full"></div>
      <div className="h-3 bg-gray-700 rounded w-5/6"></div>
    </div>
  </div>
));
GasPriceCardSkeleton.displayName = 'GasPriceCardSkeleton';

export const LeaderboardSkeleton = memo(() => (
  <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 animate-pulse">
    <div className="h-6 bg-gray-700 rounded w-48 mb-4"></div>
    <div className="space-y-3">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-md">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 bg-gray-600 rounded-full"></div>
            <div className="h-4 bg-gray-600 rounded w-32"></div>
          </div>
          <div className="h-4 bg-gray-600 rounded w-20"></div>
        </div>
      ))}
    </div>
  </div>
));
LeaderboardSkeleton.displayName = 'LeaderboardSkeleton';

export const GraphSkeleton = memo(() => (
  <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 animate-pulse">
    <div className="h-6 bg-gray-700 rounded w-40 mb-6"></div>
    <div className="h-64 bg-gray-700 rounded w-full"></div>
  </div>
));
GraphSkeleton.displayName = 'GraphSkeleton';

// Full Dashboard skeleton for initial page load
export const DashboardSkeleton = memo(() => (
  <div className="min-h-screen bg-gray-900 p-4">
    {/* Header skeleton */}
    <div className="max-w-7xl mx-auto">
      <div className="bg-gray-800 rounded-lg p-4 mb-6 animate-pulse">
        <div className="flex justify-between items-center">
          <div className="h-8 bg-gray-700 rounded w-32"></div>
          <div className="h-8 bg-gray-700 rounded w-24"></div>
        </div>
      </div>
      
      {/* Main content skeleton */}
      <div className="bg-gray-800 rounded-2xl p-6 mb-6 animate-pulse">
        <div className="h-8 bg-gray-700 rounded w-48 mb-4"></div>
        <div className="h-32 bg-gray-700 rounded w-full"></div>
      </div>
      
      {/* Grid skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-2xl p-6 animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-40 mb-4"></div>
          <div className="h-48 bg-gray-700 rounded"></div>
        </div>
        <div className="bg-gray-800 rounded-2xl p-6 animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-40 mb-4"></div>
          <div className="h-48 bg-gray-700 rounded"></div>
        </div>
      </div>
    </div>
  </div>
));
DashboardSkeleton.displayName = 'DashboardSkeleton';

export const ErrorFallback = ({
  error,
  onRetry
}: {
  error?: string;
  onRetry?: () => void;
}) => (
  <div className="bg-gray-800 border border-yellow-500/30 rounded-lg p-6">
    <div className="flex items-center gap-3 mb-3">
      <span className="text-2xl">⚠️</span>
      <h3 className="text-lg font-semibold text-yellow-400">Unable to load data</h3>
    </div>
    <p className="text-gray-400 text-sm mb-4">
      {error || "We're having trouble fetching the latest data. This might be temporary."}
    </p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg text-sm font-medium transition-colors"
      >
        Try Again
      </button>
    )}
  </div>
);

export default {
  CardSkeleton,
  GasPriceCardSkeleton,
  LeaderboardSkeleton,
  GraphSkeleton,
  DashboardSkeleton,
  ErrorFallback
};
