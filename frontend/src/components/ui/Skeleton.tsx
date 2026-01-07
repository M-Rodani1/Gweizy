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
  <div className={`bg-gray-800/50 rounded-xl p-4 space-y-3 ${className}`}>
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
  <div className={`bg-gray-800/50 rounded-xl p-4 space-y-4 ${className}`}>
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
  <div className={`bg-gray-800/50 rounded-xl p-4 space-y-4 ${className}`}>
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
  <div className="bg-gray-800/50 rounded-xl overflow-hidden">
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

export default Skeleton;
