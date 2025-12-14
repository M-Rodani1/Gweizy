import React, { useEffect, useState } from 'react';

interface PriceLevel {
  level: string;
  color: string;
  emoji: string;
  description: string;
  action: string;
}

interface RelativePriceIndicatorProps {
  currentGas: number;
  className?: string;
}

const RelativePriceIndicator: React.FC<RelativePriceIndicatorProps> = ({
  currentGas,
  className = ''
}) => {
  const [hourlyAverage, setHourlyAverage] = useState<number>(0);
  const [dayAverage, setDayAverage] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAverages = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || 'https://basegasfeesml.onrender.com/api'}/historical?hours=24`);
        const data = await response.json();

        if (data.historical && data.historical.length > 0) {
          const currentHour = new Date().getUTCHours();

          // Calculate current hour average
          const hourData = data.historical.filter((point: any) => {
            const timestamp = new Date(point.timestamp);
            return timestamp.getUTCHours() === currentHour;
          });

          if (hourData.length > 0) {
            const hourAvg = hourData.reduce((sum: number, p: any) =>
              sum + (p.gwei || p.current_gas || 0), 0) / hourData.length;
            setHourlyAverage(hourAvg);
          }

          // Calculate 24h average
          const dayAvg = data.historical.reduce((sum: number, p: any) =>
            sum + (p.gwei || p.current_gas || 0), 0) / data.historical.length;
          setDayAverage(dayAvg);
        }
      } catch (error) {
        console.error('Failed to fetch averages:', error);
        // Fallback to reasonable defaults
        setHourlyAverage(0.0024);
        setDayAverage(0.0024);
      } finally {
        setLoading(false);
      }
    };

    fetchAverages();

    // Refresh every 5 minutes
    const interval = setInterval(fetchAverages, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const getPriceLevel = (current: number, avg: number): PriceLevel => {
    const ratio = current / avg;

    if (ratio < 0.7) {
      return {
        level: 'Excellent',
        color: 'green',
        emoji: '🟢',
        description: 'Gas is significantly below average',
        action: 'Perfect time to transact!'
      };
    } else if (ratio < 0.9) {
      return {
        level: 'Good',
        color: 'cyan',
        emoji: '🔵',
        description: 'Gas is below average',
        action: 'Good time to transact'
      };
    } else if (ratio < 1.15) {
      return {
        level: 'Average',
        color: 'yellow',
        emoji: '🟡',
        description: 'Gas is near average',
        action: 'Typical price for this time'
      };
    } else if (ratio < 1.5) {
      return {
        level: 'High',
        color: 'orange',
        emoji: '🟠',
        description: 'Gas is above average',
        action: 'Consider waiting if not urgent'
      };
    } else {
      return {
        level: 'Very High',
        color: 'red',
        emoji: '🔴',
        description: 'Gas is significantly above average',
        action: 'Wait unless transaction is urgent'
      };
    }
  };

  if (loading || !currentGas) {
    return (
      <div className={`bg-gray-800 p-6 rounded-lg shadow-lg ${className}`}>
        <div className="animate-pulse flex flex-col items-center">
          <div className="w-24 h-24 bg-gray-700 rounded-full mb-4"></div>
          <div className="h-6 bg-gray-700 rounded w-32 mb-2"></div>
          <div className="h-4 bg-gray-700 rounded w-48"></div>
        </div>
      </div>
    );
  }

  const status = getPriceLevel(currentGas, hourlyAverage || dayAverage);
  const savingsVsHigh = hourlyAverage > 0 ?
    Math.round(((hourlyAverage * 1.5 - currentGas) / (hourlyAverage * 1.5)) * 100) : 0;

  return (
    <div className={`bg-gradient-to-br from-gray-800 to-gray-900 p-4 sm:p-6 rounded-lg shadow-lg border-2 border-${status.color}-500/30 ${className}`}>
      {/* Status Indicator */}
      <div className="text-center mb-4">
        <div className="text-6xl sm:text-7xl mb-3 animate-pulse">
          {status.emoji}
        </div>
        <div className={`text-xl sm:text-2xl font-bold text-${status.color}-400 mb-2`}>
          {status.level} Time to Transact
        </div>
        <div className="text-sm sm:text-base text-gray-400">
          {status.description}
        </div>
      </div>

      {/* Price Comparison */}
      <div className="space-y-2 mb-4">
        <div className="flex justify-between items-center p-2 bg-gray-700/50 rounded">
          <span className="text-xs sm:text-sm text-gray-400">Current Gas:</span>
          <span className="text-sm sm:text-base font-bold text-gray-100">
            {(currentGas * 1000).toFixed(3)} gwei
          </span>
        </div>

        {hourlyAverage > 0 && (
          <div className="flex justify-between items-center p-2 bg-gray-700/50 rounded">
            <span className="text-xs sm:text-sm text-gray-400">Hour Average:</span>
            <span className="text-sm sm:text-base font-mono text-gray-300">
              {(hourlyAverage * 1000).toFixed(3)} gwei
            </span>
          </div>
        )}

        {dayAverage > 0 && (
          <div className="flex justify-between items-center p-2 bg-gray-700/50 rounded">
            <span className="text-xs sm:text-sm text-gray-400">24h Average:</span>
            <span className="text-sm sm:text-base font-mono text-gray-300">
              {(dayAverage * 1000).toFixed(3)} gwei
            </span>
          </div>
        )}
      </div>

      {/* Action Recommendation */}
      <div className={`p-3 rounded-lg bg-${status.color}-500/10 border border-${status.color}-500/30`}>
        <div className="flex items-start gap-2">
          <span className="text-lg flex-shrink-0">{status.emoji}</span>
          <div className="flex-1 min-w-0">
            <div className={`text-xs sm:text-sm font-semibold text-${status.color}-400 mb-1`}>
              Recommendation
            </div>
            <div className="text-xs sm:text-sm text-gray-300">
              {status.action}
            </div>
            {savingsVsHigh > 0 && status.level !== 'Very High' && (
              <div className="text-xs text-gray-400 mt-1">
                Save ~{savingsVsHigh}% vs peak hours
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Price Trend (optional enhancement) */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="flex items-center justify-between text-xs text-gray-400">
          <span>Updated just now</span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            Live
          </span>
        </div>
      </div>
    </div>
  );
};

export default RelativePriceIndicator;
