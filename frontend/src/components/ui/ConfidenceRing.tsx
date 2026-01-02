import React from 'react';

interface ConfidenceRingProps {
  confidence: number; // 0-1
  size?: number;
  strokeWidth?: number;
  color?: string;
}

const ConfidenceRing: React.FC<ConfidenceRingProps> = ({
  confidence,
  size = 60,
  strokeWidth = 6,
  color
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (confidence * circumference);

  // Determine color based on confidence level
  const getColor = () => {
    if (color) return color;
    if (confidence >= 0.7) return '#22c55e'; // green
    if (confidence >= 0.4) return '#eab308'; // yellow
    return '#ef4444'; // red
  };

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        className="confidence-ring"
        width={size}
        height={size}
      >
        {/* Background circle */}
        <circle
          className="bg"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
        />
        {/* Progress circle */}
        <circle
          className="progress"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          stroke={getColor()}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ '--target-offset': offset } as React.CSSProperties}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-bold text-white">
          {Math.round(confidence * 100)}%
        </span>
      </div>
    </div>
  );
};

export default ConfidenceRing;
