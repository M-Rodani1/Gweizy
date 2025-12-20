import React from 'react';

export const CardSkeleton = () => (
  <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 animate-pulse">
    <div className="h-6 bg-gray-700 rounded w-3/4 mb-4"></div>
    <div className="h-4 bg-gray-700 rounded w-1/2 mb-2"></div>
    <div className="h-4 bg-gray-700 rounded w-2/3"></div>
  </div>
);

export const GasPriceCardSkeleton = () => (
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
);

export const LeaderboardSkeleton = () => (
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
);

export const GraphSkeleton = () => (
  <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 animate-pulse">
    <div className="h-6 bg-gray-700 rounded w-40 mb-6"></div>
    <div className="h-64 bg-gray-700 rounded w-full"></div>
  </div>
);

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
  ErrorFallback
};
