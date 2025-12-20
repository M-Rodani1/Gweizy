/**
 * Last Updated Indicator Component
 * Shows when data was last refreshed
 */

import React from 'react';
import { formatRelativeTime } from '../utils/formatters';

interface LastUpdatedProps {
  timestamp: number | null;
  className?: string;
}

export const LastUpdated: React.FC<LastUpdatedProps> = ({ 
  timestamp, 
  className = '' 
}) => {
  if (!timestamp) return null;

  return (
    <div className={`text-xs text-gray-400 flex items-center gap-1 ${className}`}>
      <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
      <span>Updated {formatRelativeTime(Math.floor(timestamp / 1000))}</span>
    </div>
  );
};
