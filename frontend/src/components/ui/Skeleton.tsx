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

export default Skeleton;
