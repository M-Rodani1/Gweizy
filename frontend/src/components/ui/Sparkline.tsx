import React, { useMemo } from 'react';

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  strokeWidth?: number;
  className?: string;
  /** Show gradient fill under the line */
  showGradient?: boolean;
  /** Show dot at the end of the line */
  showDot?: boolean;
  /** Use trend-based colors (red for up/bad, green for down/good) */
  useTrendColor?: boolean;
}

const Sparkline: React.FC<SparklineProps> = ({
  data,
  width = 60,
  height = 20,
  color = '#06b6d4',
  strokeWidth = 1.5,
  className = '',
  showGradient = false,
  showDot = false,
  useTrendColor = false
}) => {
  const { linePath, areaPath, lastPoint, effectiveColor } = useMemo(() => {
    if (!data || data.length < 2) {
      return { linePath: '', areaPath: '', lastPoint: null, effectiveColor: color };
    }

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

    // Create filled area path for gradient
    const areaPath = showGradient
      ? `${linePath} L ${width},${height} L 0,${height} Z`
      : '';

    // Last point for dot
    const lastPoint = points[points.length - 1];

    // Trend color: red for up (bad for gas), green for down (good)
    const isUp = data[data.length - 1] > data[0];
    const trendColor = isUp ? '#ef4444' : '#22c55e';
    const effectiveColor = useTrendColor ? trendColor : color;

    return { linePath, areaPath, lastPoint, effectiveColor };
  }, [data, width, height, color, showGradient, useTrendColor]);

  // Empty state
  if (!data || data.length < 2) {
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

  // Generate unique gradient ID to avoid conflicts
  const gradientId = `sparkline-gradient-${effectiveColor.replace('#', '')}`;

  return (
    <div className={`inline-block ${className}`} style={{ width, height }}>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="overflow-visible"
      >
        {showGradient && (
          <defs>
            <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={effectiveColor} stopOpacity="0.3" />
              <stop offset="100%" stopColor={effectiveColor} stopOpacity="0" />
            </linearGradient>
          </defs>
        )}

        {/* Gradient fill area */}
        {showGradient && areaPath && (
          <path
            d={areaPath}
            fill={`url(#${gradientId})`}
          />
        )}

        {/* Line */}
        <path
          d={linePath}
          fill="none"
          stroke={effectiveColor}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* End dot */}
        {showDot && lastPoint && (
          <circle
            cx={lastPoint.x}
            cy={lastPoint.y}
            r="2"
            fill={effectiveColor}
          />
        )}
      </svg>
    </div>
  );
};

export default React.memo(Sparkline);
