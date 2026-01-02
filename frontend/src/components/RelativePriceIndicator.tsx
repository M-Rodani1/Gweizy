import React, { useEffect, useState } from 'react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface PriceLevel {
  level: string;
  color: string;
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
  const [weeklyAverage, setWeeklyAverage] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAverages = async () => {
      try {
        // Fetch 24h data for hourly and daily averages
        const dayResponse = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.HISTORICAL, { hours: 24 }));
        const dayData = await dayResponse.json();

        if (dayData && dayData.historical && Array.isArray(dayData.historical) && dayData.historical.length > 0) {
          const currentHour = new Date().getUTCHours();

          // Calculate current hour average
          const hourData = dayData.historical.filter((point: any) => {
            const timestamp = new Date(point.timestamp);
            return timestamp.getUTCHours() === currentHour;
          });

          if (hourData.length > 0) {
            const hourAvg = hourData.reduce((sum: number, p: any) =>
              sum + (p.gwei || p.current_gas || 0), 0) / hourData.length;
            setHourlyAverage(hourAvg);
          }

          // Calculate 24h average
          const dayAvg = dayData.historical.reduce((sum: number, p: any) =>
            sum + (p.gwei || p.current_gas || 0), 0) / dayData.historical.length;
          setDayAverage(dayAvg);
        }

        // Fetch 7-day data for weekly average
        try {
          const weekResponse = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.HISTORICAL, { hours: 168 }));
          const weekData = await weekResponse.json();

          if (weekData && weekData.historical && Array.isArray(weekData.historical) && weekData.historical.length > 0) {
            const weekAvg = weekData.historical.reduce((sum: number, p: any) =>
              sum + (p.gwei || p.current_gas || 0), 0) / weekData.historical.length;
            setWeeklyAverage(weekAvg);
          }
        } catch (weekErr) {
          console.error('Failed to fetch weekly data:', weekErr);
          // Weekly average is optional, continue without it
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
    // If gas is already extremely low in absolute terms, always show as excellent
    // regardless of relative comparison (prevents "Very High" when gas is 0.0005 gwei)
    if (current < 0.05) {
      return {
        level: 'Excellent',
        color: 'green',
        description: 'Gas is extremely low',
        action: 'Perfect time to transact!'
      };
    }

    // Prevent division by zero or Infinity
    const ratio = avg === 0 ? (current > 0 ? Infinity : 0) : current / avg;

    if (ratio < 0.7) {
      return {
        level: 'Excellent',
        color: 'green',
        description: 'Gas is significantly below average',
        action: 'Perfect time to transact!'
      };
    } else if (ratio < 0.9) {
      return {
        level: 'Good',
        color: 'cyan',
        description: 'Gas is below average',
        action: 'Good time to transact'
      };
    } else if (ratio < 1.15) {
      return {
        level: 'Average',
        color: 'yellow',
        description: 'Gas is near average',
        action: 'Typical price for this time'
      };
    } else if (ratio < 1.5) {
      return {
        level: 'High',
        color: 'orange',
        description: 'Gas is above average',
        action: 'Consider waiting if not urgent'
      };
    } else {
      return {
        level: 'Very High',
        color: 'red',
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
  // Prevent division by zero
  const divisor = hourlyAverage * 1.5;
  const savingsVsHigh = divisor > 0 ?
    Math.round(((divisor - currentGas) / divisor) * 100) : 0;

  // Map colors to actual Tailwind classes
  const colorClasses = {
    green: {
      border: 'border-green-500/30',
      text: 'text-green-400',
      bg: 'bg-green-500/10',
      borderFull: 'border-green-500/30',
      dot: 'bg-green-400'
    },
    cyan: {
      border: 'border-cyan-500/30',
      text: 'text-cyan-400',
      bg: 'bg-cyan-500/10',
      borderFull: 'border-cyan-500/30',
      dot: 'bg-cyan-400'
    },
    yellow: {
      border: 'border-yellow-500/30',
      text: 'text-yellow-400',
      bg: 'bg-yellow-500/10',
      borderFull: 'border-yellow-500/30',
      dot: 'bg-yellow-400'
    },
    orange: {
      border: 'border-orange-500/30',
      text: 'text-orange-400',
      bg: 'bg-orange-500/10',
      borderFull: 'border-orange-500/30',
      dot: 'bg-orange-400'
    },
    red: {
      border: 'border-red-500/30',
      text: 'text-red-400',
      bg: 'bg-red-500/10',
      borderFull: 'border-red-500/30',
      dot: 'bg-red-400'
    }
  };

  const colors = colorClasses[status.color as keyof typeof colorClasses] || colorClasses.yellow;

  return (
    <div className={`bg-gradient-to-br from-gray-800 to-gray-900 p-4 sm:p-6 rounded-2xl shadow-2xl border-2 ${colors.border} card-hover ${className}`}>
      {/* Status Indicator */}
      <div className="text-center mb-4">
        <div className="mx-auto mb-3 flex h-16 w-16 items-center justify-center rounded-full border border-gray-700 bg-gray-900/60">
          <span className={`h-4 w-4 rounded-full ${colors.dot} animate-pulse`} />
        </div>
        <div className={`text-xl sm:text-2xl font-bold ${colors.text} mb-2`}>
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
            {currentGas !== undefined && currentGas !== null ? currentGas.toFixed(4) : 'N/A'} gwei
          </span>
        </div>

        {hourlyAverage > 0 && (
          <div className="flex justify-between items-center p-2 bg-gray-700/50 rounded">
            <span className="text-xs sm:text-sm text-gray-400">Hour Average:</span>
            <span className="text-sm sm:text-base font-mono text-gray-300">
              {hourlyAverage !== undefined && hourlyAverage !== null ? hourlyAverage.toFixed(4) : 'N/A'} gwei
            </span>
          </div>
        )}

        {dayAverage > 0 && (
          <div className="flex justify-between items-center p-2 bg-gray-700/50 rounded">
            <span className="text-xs sm:text-sm text-gray-400">24h Average:</span>
            <span className="text-sm sm:text-base font-mono text-gray-300">
              {dayAverage !== undefined && dayAverage !== null ? dayAverage.toFixed(4) : 'N/A'} gwei
            </span>
          </div>
        )}

        {weeklyAverage > 0 && (
          <div className="flex justify-between items-center p-2 bg-gray-700/50 rounded">
            <span className="text-xs sm:text-sm text-gray-400">7-Day Average:</span>
            <div className="text-right">
              <span className="text-sm sm:text-base font-mono text-gray-300">
                {weeklyAverage.toFixed(4)} gwei
              </span>
              {currentGas > 0 && weeklyAverage > 0 && (
                <div className={`text-xs ${currentGas < weeklyAverage ? 'text-green-400' : currentGas > weeklyAverage ? 'text-red-400' : 'text-gray-400'}`}>
                  {currentGas < weeklyAverage
                    ? `${(((weeklyAverage - currentGas) / weeklyAverage) * 100).toFixed(0)}% below avg`
                    : currentGas > weeklyAverage
                    ? `${(((currentGas - weeklyAverage) / weeklyAverage) * 100).toFixed(0)}% above avg`
                    : 'At average'}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Action Recommendation */}
      <div className={`p-3 rounded-lg ${colors.bg} border ${colors.borderFull}`}>
        <div className="flex items-start gap-2">
          <span className={`mt-1 h-2.5 w-2.5 rounded-full ${colors.dot}`} />
          <div className="flex-1 min-w-0">
            <div className={`text-xs sm:text-sm font-semibold ${colors.text} mb-1`}>
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
