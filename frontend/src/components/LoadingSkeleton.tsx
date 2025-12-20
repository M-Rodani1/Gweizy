/**
 * Loading skeleton components
 * Provides better UX than generic spinners
 */

import React from 'react';

interface SkeletonProps {
  className?: string;
  width?: string | number;
  height?: string | number;
}

/**
 * Base skeleton component
 */
export const Skeleton: React.FC<SkeletonProps> = ({ 
  className = '', 
  width, 
  height 
}) => {
  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  return (
    <div
      className={`bg-gray-700 rounded animate-pulse ${className}`}
      style={style}
    />
  );
};

/**
 * Card skeleton
 */
export const CardSkeleton: React.FC = () => (
  <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
    <Skeleton height={24} width="60%" className="mb-4" />
    <Skeleton height={16} width="80%" className="mb-2" />
    <Skeleton height={16} width="40%" />
  </div>
);

/**
 * Table skeleton
 */
export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 5 }) => (
  <div className="bg-gray-800 p-6 rounded-lg">
    <Skeleton height={32} width="100%" className="mb-4" />
    {Array.from({ length: rows }).map((_, i) => (
      <div key={i} className="flex gap-4 mb-3">
        <Skeleton height={20} width="30%" />
        <Skeleton height={20} width="20%" />
        <Skeleton height={20} width="25%" />
        <Skeleton height={20} width="25%" />
      </div>
    ))}
  </div>
);

/**
 * Graph skeleton
 */
export const GraphSkeleton: React.FC = () => (
  <div className="bg-gray-800 p-6 rounded-lg">
    <Skeleton height={24} width="40%" className="mb-4" />
    <Skeleton height={300} width="100%" />
  </div>
);

/**
 * Heatmap skeleton
 */
export const HeatmapSkeleton: React.FC = () => (
  <div className="bg-gray-800 p-6 rounded-lg">
    <Skeleton height={24} width="50%" className="mb-4" />
    <div className="grid grid-cols-12 gap-1">
      {Array.from({ length: 24 }).map((_, i) => (
        <Skeleton key={i} className="aspect-square rounded" />
      ))}
    </div>
  </div>
);
