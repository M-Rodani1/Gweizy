/**
 * Memoized version of RelativePriceIndicator
 * Optimized to prevent unnecessary re-renders
 */

import React, { memo } from 'react';
import RelativePriceIndicator from './RelativePriceIndicator';

interface MemoizedRelativePriceIndicatorProps {
  currentGas: number;
  className?: string;
}

/**
 * Memoized RelativePriceIndicator component
 * Only re-renders when currentGas or className changes
 */
export const MemoizedRelativePriceIndicator = memo<MemoizedRelativePriceIndicatorProps>(
  ({ currentGas, className }) => {
    return <RelativePriceIndicator currentGas={currentGas} className={className} />;
  },
  (prevProps, nextProps) => {
    // Custom comparison: only re-render if currentGas changes significantly (> 1%)
    const prevGas = prevProps.currentGas;
    const nextGas = nextProps.currentGas;
    const gasDiff = Math.abs((nextGas - prevGas) / prevGas);
    
    return (
      gasDiff < 0.01 && // Less than 1% change
      prevProps.className === nextProps.className
    );
  }
);

MemoizedRelativePriceIndicator.displayName = 'MemoizedRelativePriceIndicator';
