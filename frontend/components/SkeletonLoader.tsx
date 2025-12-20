import React from 'react';

interface SkeletonLoaderProps {
  type?: 'card' | 'chart' | 'table' | 'text' | 'metric';
  count?: number;
  className?: string;
}

const SkeletonLoader: React.FC<SkeletonLoaderProps> = ({ type = 'card', count = 1, className = '' }) => {
  const renderSkeleton = () => {
    switch (type) {
      case 'card':
        return (
          <div className={`bg-gray-800 rounded-xl p-6 border border-gray-700 ${className}`}>
            <div className="shimmer h-6 bg-gray-700 rounded w-1/3 mb-4"></div>
            <div className="space-y-3">
              <div className="shimmer h-4 bg-gray-700 rounded w-full"></div>
              <div className="shimmer h-4 bg-gray-700 rounded w-5/6"></div>
              <div className="shimmer h-4 bg-gray-700 rounded w-4/6"></div>
            </div>
          </div>
        );

      case 'chart':
        return (
          <div className={`bg-gray-800 rounded-xl p-6 border border-gray-700 ${className}`}>
            <div className="shimmer h-6 bg-gray-700 rounded w-1/4 mb-6"></div>
            <div className="flex items-end justify-between h-48 gap-2">
              {[...Array(12)].map((_, i) => (
                <div
                  key={i}
                  className="shimmer bg-gray-700 rounded-t w-full"
                  style={{ height: `${Math.random() * 80 + 20}%` }}
                ></div>
              ))}
            </div>
          </div>
        );

      case 'table':
        return (
          <div className={`bg-gray-800 rounded-xl overflow-hidden border border-gray-700 ${className}`}>
            <div className="p-4 border-b border-gray-700">
              <div className="shimmer h-5 bg-gray-700 rounded w-1/4"></div>
            </div>
            {[...Array(5)].map((_, i) => (
              <div key={i} className="flex items-center gap-4 p-4 border-b border-gray-700/50">
                <div className="shimmer h-4 bg-gray-700 rounded w-1/6"></div>
                <div className="shimmer h-4 bg-gray-700 rounded w-1/4"></div>
                <div className="shimmer h-4 bg-gray-700 rounded w-1/5"></div>
                <div className="shimmer h-4 bg-gray-700 rounded w-1/6"></div>
              </div>
            ))}
          </div>
        );

      case 'metric':
        return (
          <div className={`bg-gray-800 rounded-xl p-6 border border-gray-700 ${className}`}>
            <div className="shimmer h-4 bg-gray-700 rounded w-1/3 mb-3"></div>
            <div className="shimmer h-10 bg-gray-700 rounded w-2/3 mb-2"></div>
            <div className="shimmer h-3 bg-gray-700 rounded w-1/2"></div>
          </div>
        );

      case 'text':
        return (
          <div className={`space-y-2 ${className}`}>
            <div className="shimmer h-4 bg-gray-700 rounded w-full"></div>
            <div className="shimmer h-4 bg-gray-700 rounded w-5/6"></div>
            <div className="shimmer h-4 bg-gray-700 rounded w-4/6"></div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <>
      {[...Array(count)].map((_, index) => (
        <div key={index} className="animate-pulse">
          {renderSkeleton()}
        </div>
      ))}
    </>
  );
};

export default SkeletonLoader;
