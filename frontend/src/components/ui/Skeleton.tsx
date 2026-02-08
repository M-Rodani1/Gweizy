import React from 'react';

interface SkeletonProps {
  variant?: 'text' | 'card' | 'circle' | 'rect';
  width?: string | number;
  height?: string | number;
  className?: string;
}

const Skeleton: React.FC<SkeletonProps> = ({
  variant = 'rect',
  width,
  height,
  className = ''
}) => {
  const baseClasses = 'skeleton animate-pulse';

  const variantClasses = {
    text: 'skeleton-text h-4 rounded',
    card: 'skeleton-card rounded-xl',
    circle: 'rounded-full',
    rect: 'rounded-lg'
  };

  const style: React.CSSProperties = {
    width: typeof width === 'number' ? `${width}px` : width,
    height: typeof height === 'number' ? `${height}px` : height
  };

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
    />
  );
};

// Pre-built skeleton layouts
export const SkeletonCard: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-gray-800/50 rounded-2xl p-6 space-y-3 ${className}`}>
    <div className="flex items-center gap-3">
      <Skeleton variant="circle" width={40} height={40} />
      <div className="flex-1 space-y-2">
        <Skeleton variant="text" width="60%" />
        <Skeleton variant="text" width="40%" />
      </div>
    </div>
    <Skeleton variant="rect" height={100} className="w-full" />
    <div className="flex gap-2">
      <Skeleton variant="rect" width="30%" height={32} />
      <Skeleton variant="rect" width="30%" height={32} />
    </div>
  </div>
);

export const SkeletonList: React.FC<{ count?: number }> = ({ count = 3 }) => (
  <div className="space-y-3">
    {Array.from({ length: count }).map((_, i) => (
      <div key={i} className="flex items-center gap-3 p-3">
        <Skeleton variant="circle" width={32} height={32} />
        <div className="flex-1 space-y-2">
          <Skeleton variant="text" width="70%" />
          <Skeleton variant="text" width="50%" />
        </div>
        <Skeleton variant="rect" width={60} height={24} />
      </div>
    ))}
  </div>
);

export const SkeletonMetrics: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-gray-800/50 rounded-2xl p-6 space-y-4 ${className}`}>
    <div className="flex items-center justify-between">
      <Skeleton variant="text" width={120} height={20} />
      <Skeleton variant="rect" width={80} height={28} />
    </div>
    <div className="grid grid-cols-2 gap-3">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="bg-gray-900/50 rounded-lg p-3 space-y-2">
          <Skeleton variant="text" width="60%" height={12} />
          <Skeleton variant="text" width="80%" height={24} />
          <Skeleton variant="text" width="50%" height={10} />
        </div>
      ))}
    </div>
  </div>
);

export const SkeletonChart: React.FC<{ className?: string; height?: number }> = ({
  className,
  height = 200,
}) => (
  <div className={`bg-gray-800/50 rounded-2xl p-6 space-y-4 ${className}`}>
    <div className="flex items-center justify-between">
      <Skeleton variant="text" width={150} height={20} />
      <div className="flex gap-2">
        <Skeleton variant="rect" width={60} height={24} />
        <Skeleton variant="rect" width={60} height={24} />
      </div>
    </div>
    <Skeleton variant="rect" height={height} className="w-full" />
  </div>
);

export const SkeletonTable: React.FC<{ rows?: number; cols?: number }> = ({
  rows = 5,
  cols = 4,
}) => (
  <div className="bg-gray-800/50 rounded-2xl overflow-hidden">
    {/* Header */}
    <div className="bg-gray-900/50 px-4 py-3 flex gap-4">
      {Array.from({ length: cols }).map((_, i) => (
        <Skeleton key={i} variant="text" width={`${100 / cols}%`} height={16} />
      ))}
    </div>
    {/* Rows */}
    {Array.from({ length: rows }).map((_, rowIdx) => (
      <div key={rowIdx} className="px-4 py-3 flex gap-4 border-t border-gray-700/50">
        {Array.from({ length: cols }).map((_, colIdx) => (
          <Skeleton key={colIdx} variant="text" width={`${100 / cols}%`} height={14} />
        ))}
      </div>
    ))}
  </div>
);

// ============================================================================
// Application-Specific Skeletons
// ============================================================================

/**
 * Skeleton for gas prediction card with big number display
 */
export const SkeletonGasPrediction: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-gray-800/50 rounded-2xl p-6 space-y-4 border border-gray-700/50 ${className}`}>
    {/* Header */}
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <Skeleton variant="circle" width={24} height={24} />
        <Skeleton variant="text" width={100} height={16} />
      </div>
      <Skeleton variant="rect" width={50} height={20} className="rounded-full" />
    </div>
    {/* Big number */}
    <div className="py-3">
      <Skeleton variant="text" width="60%" height={48} className="mx-auto" />
      <Skeleton variant="text" width="40%" height={14} className="mx-auto mt-2" />
    </div>
    {/* Bottom info */}
    <div className="flex justify-between pt-3 border-t border-gray-700/30">
      <Skeleton variant="text" width={80} height={12} />
      <Skeleton variant="text" width={60} height={12} />
    </div>
  </div>
);

/**
 * Skeleton for accuracy metrics dashboard
 */
export const SkeletonAccuracyMetrics: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-slate-800 rounded-2xl p-6 border border-slate-700 space-y-6 ${className}`}>
    {/* Header with tabs */}
    <div className="flex items-center justify-between">
      <Skeleton variant="text" width={200} height={24} />
      <div className="flex gap-2">
        <Skeleton variant="rect" width={40} height={28} className="rounded-lg" />
        <Skeleton variant="rect" width={40} height={28} className="rounded-lg" />
        <Skeleton variant="rect" width={40} height={28} className="rounded-lg" />
      </div>
    </div>
    {/* Metrics cards */}
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="bg-slate-900 rounded-2xl p-6 border border-slate-700 space-y-2">
          <Skeleton variant="text" width="70%" height={14} />
          <Skeleton variant="text" width="50%" height={32} />
          <Skeleton variant="text" width="60%" height={10} />
        </div>
      ))}
    </div>
    {/* Chart area */}
    <div className="bg-slate-900 rounded-2xl p-6 border border-slate-700">
      <Skeleton variant="text" width={180} height={18} className="mb-4" />
      <Skeleton variant="rect" height={200} className="w-full" />
    </div>
  </div>
);

/**
 * Skeleton for multi-chain comparison list
 */
export const SkeletonMultiChain: React.FC<{ className?: string; count?: number }> = ({
  className,
  count = 5
}) => (
  <div className={`bg-gray-800/50 border border-gray-700 rounded-2xl overflow-hidden ${className}`}>
    {/* Header */}
    <div className="px-6 py-4 border-b border-gray-700/50 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <Skeleton variant="circle" width={16} height={16} />
        <Skeleton variant="text" width={120} height={18} />
      </div>
      <Skeleton variant="rect" width={60} height={20} className="rounded" />
    </div>
    {/* Chain list */}
    <div className="divide-y divide-gray-700/30">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Skeleton variant="circle" width={28} height={28} />
            <div className="flex items-center gap-2">
              <Skeleton variant="circle" width={24} height={24} />
              <div className="space-y-1">
                <Skeleton variant="text" width={80} height={14} />
                <Skeleton variant="text" width={50} height={10} />
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Skeleton variant="rect" width={48} height={20} className="hidden sm:block" />
            <div className="text-right space-y-1">
              <Skeleton variant="text" width={55} height={16} />
              <Skeleton variant="text" width={35} height={10} />
            </div>
          </div>
        </div>
      ))}
    </div>
    {/* Footer */}
    <div className="px-6 py-4 bg-gray-800/50 border-t border-gray-700/30">
      <div className="flex items-center justify-between">
        <Skeleton variant="text" width={150} height={14} />
        <Skeleton variant="text" width={80} height={14} />
      </div>
    </div>
  </div>
);

/**
 * Skeleton for gas price header/hero section
 */
export const SkeletonGasHero: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-gray-800/50 rounded-2xl p-6 border border-gray-700 ${className}`}>
    <div className="flex flex-col items-center space-y-4">
      <Skeleton variant="text" width={120} height={16} />
      <Skeleton variant="text" width={200} height={64} />
      <div className="flex items-center gap-4">
        <Skeleton variant="rect" width={80} height={24} className="rounded-full" />
        <Skeleton variant="text" width={100} height={14} />
      </div>
    </div>
  </div>
);

/**
 * Skeleton for network intelligence panel
 */
export const SkeletonNetworkIntel: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-gray-800/50 rounded-2xl p-6 border border-gray-700/50 space-y-4 ${className}`}>
    {/* Header */}
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <Skeleton variant="circle" width={20} height={20} />
        <Skeleton variant="text" width={140} height={18} />
      </div>
      <Skeleton variant="rect" width={70} height={22} className="rounded-full" />
    </div>
    {/* Stats grid */}
    <div className="grid grid-cols-2 gap-3">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="bg-gray-900/50 rounded-lg p-3 space-y-2">
          <Skeleton variant="text" width="60%" height={12} />
          <Skeleton variant="text" width="80%" height={20} />
        </div>
      ))}
    </div>
    {/* Mini chart */}
    <Skeleton variant="rect" height={80} className="w-full rounded-lg" />
  </div>
);

/**
 * Skeleton for CompactForecast component
 */
export const SkeletonForecast: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-gray-800/50 border border-gray-700 rounded-2xl overflow-hidden h-full flex flex-col ${className}`}>
    {/* Header */}
    <div className="px-6 py-4 border-b border-gray-700/50 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <Skeleton variant="circle" width={16} height={16} />
        <Skeleton variant="text" width={100} height={16} />
      </div>
      <Skeleton variant="text" width={60} height={12} />
    </div>

    {/* Current Price */}
    <div className="px-6 py-4 border-b border-gray-700/30 bg-gray-800/30">
      <div className="flex items-center justify-between">
        <Skeleton variant="text" width={50} height={14} />
        <Skeleton variant="text" width={90} height={22} />
      </div>
    </div>

    {/* Predictions */}
    <div className="divide-y divide-gray-700/30 flex-1">
      {[1, 2, 3].map((i) => (
        <div key={i} className="px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Skeleton variant="text" width={55} height={14} />
            <Skeleton variant="circle" width={16} height={16} />
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right space-y-1">
              <Skeleton variant="text" width={65} height={16} />
              <Skeleton variant="text" width={40} height={12} />
            </div>
            <div className="w-16 space-y-1">
              <Skeleton variant="rect" width={64} height={6} className="rounded-full" />
              <Skeleton variant="text" width={30} height={10} className="mx-auto" />
            </div>
          </div>
        </div>
      ))}
    </div>

    {/* Footer */}
    <div className="px-6 py-4 bg-gray-800/50 border-t border-gray-700/30">
      <Skeleton variant="text" width="60%" height={14} />
    </div>
  </div>
);

/**
 * Skeleton for ProfilePanel/PersonalizationPanel
 */
export const SkeletonProfile: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-gray-900/50 border border-gray-800 rounded-2xl p-5 h-full flex flex-col ${className}`}>
    {/* Header */}
    <div className="flex items-center gap-2 mb-4">
      <Skeleton variant="circle" width={16} height={16} />
      <Skeleton variant="text" width={110} height={14} />
    </div>

    {/* Strategy */}
    <div className="mb-4">
      <Skeleton variant="text" width={55} height={12} className="mb-2" />
      <div className="flex gap-2">
        {[1, 2, 3].map((i) => (
          <Skeleton key={i} variant="rect" height={36} className="flex-1 rounded-lg" />
        ))}
      </div>
    </div>

    {/* Transaction types */}
    <div className="mb-4">
      <Skeleton variant="text" width={110} height={12} className="mb-2" />
      <div className="flex flex-wrap gap-1.5">
        {[1, 2, 3, 4, 5].map((i) => (
          <Skeleton key={i} variant="rect" width={65} height={28} className="rounded-lg" />
        ))}
      </div>
    </div>

    {/* Urgency */}
    <div className="mb-4">
      <div className="flex justify-between mb-1.5">
        <Skeleton variant="text" width={50} height={12} />
        <Skeleton variant="text" width={35} height={12} />
      </div>
      <Skeleton variant="rect" height={6} className="rounded-full" />
    </div>

    {/* Advanced toggle */}
    <div className="mt-auto pt-2 border-t border-gray-800">
      <Skeleton variant="rect" height={28} className="rounded" />
    </div>
  </div>
);

// ============================================================================
// Heatmap Skeleton
// ============================================================================

export const SkeletonHeatmap: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`bg-gray-800/50 rounded-2xl p-6 border border-gray-700 ${className}`}>
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <Skeleton variant="circle" width={20} height={20} />
        <Skeleton variant="text" width={140} height={16} />
      </div>
      <Skeleton variant="rect" width={40} height={20} className="rounded" />
    </div>
    <div className="flex gap-2 mb-4">
      <Skeleton variant="rect" height={40} className="flex-1 rounded-lg" />
      <Skeleton variant="rect" height={40} className="flex-1 rounded-lg" />
    </div>
    <div className="space-y-1">
      {Array.from({ length: 7 }).map((_, i) => (
        <div key={i} className="flex gap-1">
          <Skeleton variant="text" width={28} height={16} />
          <div className="flex-1 flex gap-px">
            {Array.from({ length: 24 }).map((_, j) => (
              <Skeleton key={j} variant="rect" height={16} className="flex-1" />
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
);

// ============================================================================
// Error Fallback Component
// ============================================================================

interface ErrorFallbackProps {
  error?: string;
  onRetry?: () => void;
}

export const ErrorFallback: React.FC<ErrorFallbackProps> = ({ error, onRetry }) => (
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

// ============================================================================
// Simple Aliases for backwards compatibility
// ============================================================================

/** @deprecated Use SkeletonCard instead */
export const CardSkeleton = SkeletonCard;

/** @deprecated Use SkeletonChart instead */
export const GraphSkeleton = SkeletonChart;

/** @deprecated Use SkeletonChart instead */
export const ChartSkeleton = SkeletonChart;

/** @deprecated Use SkeletonHeatmap instead */
export const HeatmapSkeleton = SkeletonHeatmap;

export default Skeleton;
