import React from 'react';

interface StatProps {
  label: string;
  value: React.ReactNode;
  helper?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}

const trendColor = {
  up: 'text-green-400',
  down: 'text-red-400',
  neutral: 'text-gray-400'
};

export const Stat: React.FC<StatProps> = ({ label, value, helper, trend = 'neutral' }) => (
  <div className="landing-stat focus-card" role="article" tabIndex={0} aria-label={`${label}: ${value}`}>
    <div className="landing-stat-value">{value}</div>
    <div className="landing-stat-label">{label}</div>
    {helper && <div className={`text-xs mt-1 ${trendColor[trend]}`}>{helper}</div>}
  </div>
);

export default Stat;
