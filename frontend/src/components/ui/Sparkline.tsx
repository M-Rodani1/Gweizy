import React from 'react';

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  showDot?: boolean;
}

const Sparkline: React.FC<SparklineProps> = ({
  data,
  width = 60,
  height = 20,
  color = '#06b6d4',
  showDot = true
}) => {
  if (!data || data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  // Normalize data to fit in the height
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    return `${x},${y}`;
  });

  const pathD = `M ${points.join(' L ')}`;

  // Determine if trend is up or down
  const isUp = data[data.length - 1] > data[0];
  const trendColor = isUp ? '#ef4444' : '#22c55e'; // red for up (bad), green for down (good)

  return (
    <svg
      className="sparkline"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      <path
        d={pathD}
        stroke={color || trendColor}
        strokeWidth="1.5"
        fill="none"
      />
      {showDot && (
        <circle
          cx={width}
          cy={height - ((data[data.length - 1] - min) / range) * height}
          r="2"
          fill={color || trendColor}
        />
      )}
    </svg>
  );
};

export default Sparkline;
