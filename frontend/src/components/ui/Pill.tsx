import React from 'react';

interface PillProps {
  children: React.ReactNode;
  color?: 'cyan' | 'green' | 'yellow' | 'gray';
  className?: string;
}

const colorClass: Record<NonNullable<PillProps['color']>, string> = {
  cyan: 'bg-cyan-500/15 text-cyan-200 border border-cyan-500/30',
  green: 'bg-green-500/15 text-green-200 border border-green-500/30',
  yellow: 'bg-yellow-500/15 text-yellow-200 border border-yellow-500/30',
  gray: 'bg-gray-700/40 text-gray-200 border border-gray-600/60'
};

export const Pill: React.FC<PillProps> = ({ children, color = 'gray', className = '' }) => (
  <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold ${colorClass[color]} ${className}`}>
    {children}
  </span>
);

export default Pill;
