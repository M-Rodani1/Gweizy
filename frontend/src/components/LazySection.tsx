import React, { Suspense } from 'react';
import { useLazyLoad } from '../hooks/useLazyLoad';
import { SkeletonCard } from './ui/Skeleton';

interface LazySectionProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  rootMargin?: string;
  className?: string;
  style?: React.CSSProperties;
}

const ComponentLoader = () => <SkeletonCard />;

export const LazySection: React.FC<LazySectionProps> = ({
  children,
  fallback = <ComponentLoader />,
  rootMargin = '200px',
  className,
  style
}) => {
  const { ref, isVisible } = useLazyLoad({ rootMargin, triggerOnce: true });

  return (
    <div ref={ref} className={className} style={style}>
      {isVisible ? (
        <Suspense fallback={fallback}>
          {children}
        </Suspense>
      ) : (
        fallback
      )}
    </div>
  );
};
