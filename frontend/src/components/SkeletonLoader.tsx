import React from 'react';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'text',
  width,
  height,
  animation = 'pulse'
}) => {
  const baseClasses = 'bg-gray-700/50';

  const variantClasses = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-none',
    rounded: 'rounded-xl'
  };

  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'skeleton-wave',
    none: ''
  };

  const style: React.CSSProperties = {
    width: width || (variant === 'text' ? '100%' : undefined),
    height: height || (variant === 'text' ? '1em' : undefined)
  };

  return (
    <div
      className={`${baseClasses} ${variantClasses[variant]} ${animationClasses[animation]} ${className}`}
      style={style}
    />
  );
};

// Pre-built skeleton layouts for common components
interface CardSkeletonProps {
  rows?: number;
  showHeader?: boolean;
  showIcon?: boolean;
}

export const CardSkeleton: React.FC<CardSkeletonProps> = ({
  rows = 3,
  showHeader = true,
  showIcon = true
}) => (
  <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
    {showHeader && (
      <div className="flex items-center gap-3 mb-4">
        {showIcon && <Skeleton variant="circular" width={32} height={32} />}
        <div className="flex-1">
          <Skeleton variant="text" width="60%" height={16} className="mb-1" />
          <Skeleton variant="text" width="40%" height={12} />
        </div>
      </div>
    )}
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} variant="text" height={16} width={`${100 - i * 15}%`} />
      ))}
    </div>
  </div>
);

export const ChartSkeleton: React.FC<{ height?: number }> = ({ height = 200 }) => (
  <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
    <div className="flex items-center justify-between mb-4">
      <Skeleton variant="text" width={120} height={20} />
      <div className="flex gap-2">
        <Skeleton variant="rounded" width={60} height={24} />
        <Skeleton variant="rounded" width={60} height={24} />
      </div>
    </div>
    <Skeleton variant="rounded" height={height} className="w-full" />
    <div className="flex justify-between mt-3">
      <Skeleton variant="text" width={60} height={12} />
      <Skeleton variant="text" width={60} height={12} />
    </div>
  </div>
);

export const TableSkeleton: React.FC<{ rows?: number; cols?: number }> = ({
  rows = 5,
  cols = 4
}) => (
  <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
    <Skeleton variant="text" width={150} height={20} className="mb-4" />
    <div className="space-y-2">
      {/* Header row */}
      <div className="flex gap-4 pb-2 border-b border-gray-800">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} variant="text" height={14} className="flex-1" />
        ))}
      </div>
      {/* Data rows */}
      {Array.from({ length: rows }).map((_, rowIdx) => (
        <div key={rowIdx} className="flex gap-4 py-2">
          {Array.from({ length: cols }).map((_, colIdx) => (
            <Skeleton
              key={colIdx}
              variant="text"
              height={16}
              className="flex-1"
              width={colIdx === 0 ? '80%' : undefined}
            />
          ))}
        </div>
      ))}
    </div>
  </div>
);

export const MetricCardSkeleton: React.FC = () => (
  <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
    <div className="flex items-center gap-2 mb-3">
      <Skeleton variant="circular" width={24} height={24} />
      <Skeleton variant="text" width={80} height={14} />
    </div>
    <Skeleton variant="text" width={100} height={32} className="mb-2" />
    <Skeleton variant="text" width={60} height={12} />
  </div>
);

export const HeatmapSkeleton: React.FC = () => (
  <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <Skeleton variant="circular" width={20} height={20} />
        <Skeleton variant="text" width={140} height={16} />
      </div>
      <Skeleton variant="rounded" width={40} height={20} />
    </div>
    <div className="flex gap-2 mb-4">
      <Skeleton variant="rounded" height={40} className="flex-1" />
      <Skeleton variant="rounded" height={40} className="flex-1" />
    </div>
    <div className="space-y-1">
      {Array.from({ length: 7 }).map((_, i) => (
        <div key={i} className="flex gap-1">
          <Skeleton variant="text" width={28} height={16} />
          <div className="flex-1 flex gap-px">
            {Array.from({ length: 24 }).map((_, j) => (
              <Skeleton key={j} variant="rectangular" height={16} className="flex-1" />
            ))}
          </div>
        </div>
      ))}
    </div>
  </div>
);

export const PredictionCardSkeleton: React.FC = () => (
  <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
    <div className="flex items-center justify-between mb-3">
      <Skeleton variant="text" width={60} height={16} />
      <Skeleton variant="rounded" width={50} height={20} />
    </div>
    <Skeleton variant="text" width={120} height={36} className="mb-2" />
    <div className="flex items-center gap-2">
      <Skeleton variant="text" width={80} height={14} />
      <Skeleton variant="circular" width={16} height={16} />
    </div>
    <div className="mt-4 pt-3 border-t border-gray-800">
      <Skeleton variant="text" height={40} className="w-full" />
    </div>
  </div>
);

// Loading grid for multiple cards
export const LoadingGrid: React.FC<{
  count?: number;
  cols?: number;
  CardComponent?: React.FC;
}> = ({ count = 3, cols = 3, CardComponent = MetricCardSkeleton }) => (
  <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-${cols} gap-4`}>
    {Array.from({ length: count }).map((_, i) => (
      <CardComponent key={i} />
    ))}
  </div>
);

export default Skeleton;
