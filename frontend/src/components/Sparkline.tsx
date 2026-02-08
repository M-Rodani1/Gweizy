import React, { useMemo } from 'react';

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillColor?: string;
  strokeWidth?: number;
  className?: string;
  showGradient?: boolean;
}

const Sparkline: React.FC<SparklineProps> = ({
  data,
  width = 100,
  height = 30,
  color = '#06b6d4',
  strokeWidth = 2,
  className = '',
  showGradient = true
}) => {
  const pathData = useMemo(() => {
    if (!data || data.length === 0) return { linePath: '', areaPath: '' };

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return { x, y };
    });

    // Create line path
    const linePath = points
      .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x},${point.y}`)
      .join(' ');

    // Create filled area path
    const areaPath = showGradient
      ? `${linePath} L ${width},${height} L 0,${height} Z`
      : '';

    return { linePath, areaPath };
  }, [data, width, height, showGradient]);

  if (!data || data.length === 0) {
    return (
      <div className={`inline-block ${className}`} style={{ width, height }}>
        <svg width={width} height={height}>
          <line
            x1="0"
            y1={height / 2}
            x2={width}
            y2={height / 2}
            stroke="#374151"
            strokeWidth="1"
            strokeDasharray="2,2"
          />
        </svg>
      </div>
    );
  }

  return (
    <div className={`inline-block ${className}`} style={{ width, height }}>
      <svg width={width} height={height} className="overflow-visible">
        <defs>
          <linearGradient id={`gradient-${color}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Filled area */}
        {showGradient && pathData.areaPath && (
          <path
            d={pathData.areaPath}
            fill={`url(#gradient-${color})`}
            className="transition-all duration-300"
          />
        )}

        {/* Line */}
        <path
          d={pathData.linePath}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
          className="transition-all duration-300"
        />
      </svg>
    </div>
  );
};

export default Sparkline;
