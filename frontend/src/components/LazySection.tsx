import React, { Suspense } from 'react';
import { useLazyLoad } from '../hooks/useLazyLoad';

interface LazySectionProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  rootMargin?: string;
  className?: string;
  style?: React.CSSProperties;
}

const ComponentLoader = () => (
  <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 animate-pulse">
    <div className="h-32 bg-slate-700/50 rounded"></div>
  </div>
);

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
