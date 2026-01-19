import React from 'react';
import { ChainConfig } from '../config/chains';

type ChainBadgeSize = 'sm' | 'md' | 'lg';

interface ChainBadgeProps {
  chain: ChainConfig;
  size?: ChainBadgeSize;
  className?: string;
}

const SIZE_STYLES: Record<ChainBadgeSize, string> = {
  sm: 'w-6 h-6 text-[10px]',
  md: 'w-8 h-8 text-xs',
  lg: 'w-10 h-10 text-sm'
};

const getChainLabel = (shortName: string): string => {
  if (shortName.length <= 3) return shortName.toUpperCase();
  return shortName.slice(0, 2).toUpperCase();
};

const ChainBadge: React.FC<ChainBadgeProps> = ({ chain, size = 'md', className = '' }) => {
  const label = getChainLabel(chain.shortName);
  const gradient = `linear-gradient(135deg, ${chain.color} 0%, ${chain.color}80 100%)`;

  return (
    <div
      className={`${SIZE_STYLES[size]} ${className} rounded-full flex items-center justify-center font-semibold text-white shadow-sm`}
      style={{ background: gradient }}
      aria-label={`${chain.name} network`}
      title={chain.name}
    >
      {label}
    </div>
  );
};

export default React.memo(ChainBadge);
