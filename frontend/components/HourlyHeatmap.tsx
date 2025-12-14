import React, { useEffect, useState } from 'react';

interface HourData {
  hour: number;
  avgGas: number;
  count: number;
}

interface HourlyHeatmapProps {
  className?: string;
}

const HourlyHeatmap: React.FC<HourlyHeatmapProps> = ({ className = '' }) => {
  const [hourlyData, setHourlyData] = useState<HourData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedHour, setSelectedHour] = useState<number | null>(null);

  useEffect(() => {
    const fetchHourlyData = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || 'https://basegasfeesml.onrender.com/api'}/historical?hours=168`);
        const data = await response.json();

        if (data.historical) {
          const hourlyMap: { [key: number]: number[] } = {};

          // Group by hour
          data.historical.forEach((point: any) => {
            const timestamp = new Date(point.timestamp);
            const hour = timestamp.getUTCHours();

            if (!hourlyMap[hour]) {
              hourlyMap[hour] = [];
            }
            hourlyMap[hour].push(point.gwei || point.current_gas || 0);
          });

          // Calculate averages
          const hourData: HourData[] = [];
          for (let hour = 0; hour < 24; hour++) {
            const values = hourlyMap[hour] || [];
            const avg = values.length > 0
              ? values.reduce((a, b) => a + b, 0) / values.length
              : 0;

            hourData.push({
              hour,
              avgGas: avg,
              count: values.length
            });
          }

          setHourlyData(hourData);
        }
      } catch (error) {
        console.error('Failed to fetch hourly data:', error);
        // Use fallback pattern
        setHourlyData(getFallbackData());
      } finally {
        setLoading(false);
      }
    };

    fetchHourlyData();
  }, []);

  const getFallbackData = (): HourData[] => {
    const pattern = [
      0.0019, 0.0018, 0.0017, 0.0019, 0.0020, 0.0021,
      0.0023, 0.0025, 0.0024, 0.0019, 0.0018, 0.0020,
      0.0022, 0.0025, 0.0028, 0.0031, 0.0029, 0.0027,
      0.0030, 0.0034, 0.0038, 0.0041, 0.0043, 0.0039
    ];

    return pattern.map((avg, hour) => ({
      hour,
      avgGas: avg,
      count: 100
    }));
  };

  if (loading) {
    return (
      <div className={`bg-gray-800 p-6 rounded-lg shadow-lg ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-48 mb-4"></div>
          <div className="grid grid-cols-12 gap-1">
            {Array.from({ length: 24 }).map((_, i) => (
              <div key={i} className="aspect-square bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const minGas = Math.min(...hourlyData.map(h => h.avgGas));
  const maxGas = Math.max(...hourlyData.map(h => h.avgGas));
  const currentHour = new Date().getUTCHours();

  const getColor = (gas: number) => {
    const normalized = (gas - minGas) / (maxGas - minGas);

    // Green (cheap) to Red (expensive) gradient
    if (normalized < 0.33) {
      // Green
      const intensity = 0.3 + normalized * 2;
      return `rgba(34, 197, 94, ${intensity})`;
    } else if (normalized < 0.67) {
      // Yellow/Orange
      const intensity = 0.3 + (normalized - 0.33) * 2;
      return `rgba(251, 191, 36, ${intensity})`;
    } else {
      // Red
      const intensity = 0.3 + (normalized - 0.67) * 2;
      return `rgba(239, 68, 68, ${intensity})`;
    }
  };

  const formatHour = (hour: number) => {
    return hour.toString().padStart(2, '0');
  };

  const getTimeOfDay = (hour: number) => {
    if (hour >= 6 && hour < 12) return 'Morning';
    if (hour >= 12 && hour < 18) return 'Afternoon';
    if (hour >= 18 && hour < 22) return 'Evening';
    return 'Night';
  };

  return (
    <div className={`bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg ${className}`}>
      <div className="mb-4">
        <h3 className="text-lg sm:text-xl font-bold text-gray-100 mb-1">
          24-Hour Gas Price Pattern
        </h3>
        <p className="text-xs sm:text-sm text-gray-400">
          Average gas prices by hour (UTC) - Last 7 days
        </p>
      </div>

      {/* Heatmap Grid */}
      <div className="grid grid-cols-12 gap-1 mb-4">
        {hourlyData.map((data) => {
          const isCurrentHour = data.hour === currentHour;
          const isSelected = data.hour === selectedHour;

          return (
            <div
              key={data.hour}
              className={`aspect-square rounded cursor-pointer transition-all duration-200 hover:scale-110 hover:shadow-lg relative ${
                isCurrentHour ? 'ring-2 ring-cyan-400' : ''
              } ${
                isSelected ? 'ring-2 ring-white' : ''
              }`}
              style={{ backgroundColor: getColor(data.avgGas) }}
              onClick={() => setSelectedHour(isSelected ? null : data.hour)}
              title={`${formatHour(data.hour)}:00 - ${(data.avgGas * 1000).toFixed(3)} gwei`}
            >
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[8px] sm:text-xs font-bold text-white drop-shadow-lg">
                  {formatHour(data.hour)}
                </span>
              </div>
              {isCurrentHour && (
                <div className="absolute -top-1 -right-1 w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: getColor(minGas) }}></div>
            <span className="text-xs text-gray-400">Cheap</span>
          </div>
          <div className="w-12 h-1 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded"></div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: getColor(maxGas) }}></div>
            <span className="text-xs text-gray-400">Expensive</span>
          </div>
        </div>
        <div className="flex items-center gap-1 text-xs text-cyan-400">
          <div className="w-2 h-2 rounded-full bg-cyan-400"></div>
          <span>Now</span>
        </div>
      </div>

      {/* Selected Hour Details */}
      {selectedHour !== null && (
        <div className="p-3 sm:p-4 bg-gray-700/50 rounded-lg border border-gray-600">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-gray-400 mb-1">Time (UTC)</div>
              <div className="text-lg sm:text-xl font-bold text-gray-100">
                {formatHour(selectedHour)}:00
              </div>
              <div className="text-xs text-gray-400">
                {getTimeOfDay(selectedHour)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">Avg Gas Price</div>
              <div className="text-lg sm:text-xl font-bold text-gray-100">
                {(hourlyData[selectedHour].avgGas * 1000).toFixed(3)}
              </div>
              <div className="text-xs text-gray-400">gwei</div>
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-gray-600">
            <div className="text-xs text-gray-400">vs. 24h average</div>
            <div className={`text-sm font-semibold ${
              hourlyData[selectedHour].avgGas < (maxGas + minGas) / 2
                ? 'text-green-400'
                : 'text-red-400'
            }`}>
              {hourlyData[selectedHour].avgGas < (maxGas + minGas) / 2 ? '↓ Below' : '↑ Above'} average
              ({Math.abs(Math.round(((hourlyData[selectedHour].avgGas - (maxGas + minGas) / 2) / ((maxGas + minGas) / 2)) * 100))}%)
            </div>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-gray-700">
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Cheapest</div>
          <div className="text-sm font-bold text-green-400">
            {formatHour(hourlyData.reduce((min, h) => h.avgGas < min.avgGas ? h : min).hour)}:00
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Average</div>
          <div className="text-sm font-bold text-gray-100">
            {((hourlyData.reduce((sum, h) => sum + h.avgGas, 0) / hourlyData.length) * 1000).toFixed(3)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-gray-400 mb-1">Most Expensive</div>
          <div className="text-sm font-bold text-red-400">
            {formatHour(hourlyData.reduce((max, h) => h.avgGas > max.avgGas ? h : max).hour)}:00
          </div>
        </div>
      </div>
    </div>
  );
};

export default HourlyHeatmap;
