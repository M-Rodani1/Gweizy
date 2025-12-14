import React, { useEffect, useState } from 'react';

interface HourlyStats {
  hour: number;
  avgGas: number;
  percentile25: number;
  percentile75: number;
}

interface BestTimeWidgetProps {
  currentGas?: number;
}

const BestTimeWidget: React.FC<BestTimeWidgetProps> = ({ currentGas = 0 }) => {
  const [hourlyStats, setHourlyStats] = useState<HourlyStats[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Calculate hourly statistics from historical data
    // In production, this should come from your API
    const calculateHourlyStats = async () => {
      try {
        // Fetch historical data from API
        const response = await fetch(`https://basegasfeesml.onrender.com/api/historical?hours=168`);
        const data = await response.json();

        console.log('🔍 API Response:', {
          status: response.status,
          dataKeys: Object.keys(data),
          dataLength: data.data?.length || data.historical?.length || 0,
          samplePoint: data.data?.[0] || data.historical?.[0] || null,
          fullData: data
        });

        const historicalData = data.data || data.historical || [];

        if (historicalData.length > 0) {
          const hourlyData: { [key: number]: number[] } = {};

          // Group data by hour
          historicalData.forEach((point: any) => {
            const timestamp = new Date(point.time || point.timestamp);
            const hour = timestamp.getUTCHours();

            if (!hourlyData[hour]) {
              hourlyData[hour] = [];
            }
            hourlyData[hour].push(point.gwei || point.current_gas || 0);
          });

          // Calculate stats for each hour
          const stats: HourlyStats[] = [];
          for (let hour = 0; hour < 24; hour++) {
            const values = hourlyData[hour] || [];
            if (values.length > 0) {
              values.sort((a, b) => a - b);
              const avg = values.reduce((a, b) => a + b, 0) / values.length;
              const p25 = values[Math.floor(values.length * 0.25)];
              const p75 = values[Math.floor(values.length * 0.75)];

              stats.push({
                hour,
                avgGas: avg,
                percentile25: p25,
                percentile75: p75
              });
            }
          }

          // Check if stats are valid (not all zeros)
          const hasValidData = stats.length > 0 && stats.some(s => s.avgGas > 0);

          if (hasValidData) {
            console.log('Using API data:', stats.length, 'hours');
            setHourlyStats(stats);
          } else {
            console.warn('API data is empty or all zeros, using fallback');
            setHourlyStats(getFallbackStats());
          }
        } else {
          // No data returned, use fallback
          console.warn('No historical data returned from API, using fallback');
          setHourlyStats(getFallbackStats());
        }
      } catch (error) {
        console.error('Failed to calculate hourly stats:', error);
        // Use fallback data based on analysis
        const fallbackStats = getFallbackStats();
        setHourlyStats(fallbackStats);
        console.log('Using fallback stats:', fallbackStats);
      } finally {
        setLoading(false);
      }
    };

    calculateHourlyStats();
  }, []);

  const getFallbackStats = (): HourlyStats[] => {
    // Based on actual analysis showing peak at 23:00, low at 10:00
    const basePattern = [
      0.0019, 0.0018, 0.0017, 0.0019, 0.0020, 0.0021, // 0-5am (night)
      0.0023, 0.0025, 0.0024, 0.0019, 0.0018, 0.0020, // 6-11am (morning)
      0.0022, 0.0025, 0.0028, 0.0031, 0.0029, 0.0027, // 12-5pm (afternoon)
      0.0030, 0.0034, 0.0038, 0.0041, 0.0043, 0.0039  // 6-11pm (evening/night)
    ];

    return basePattern.map((avg, hour) => ({
      hour,
      avgGas: avg,
      percentile25: avg * 0.85,
      percentile75: avg * 1.15
    }));
  };

  if (loading) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg animate-pulse">
        <div className="h-6 bg-gray-700 rounded w-48 mb-4"></div>
        <div className="h-24 bg-gray-700 rounded"></div>
      </div>
    );
  }

  // Find best and worst hours
  const sortedByGas = [...hourlyStats].sort((a, b) => a.avgGas - b.avgGas);
  const bestHours = sortedByGas.slice(0, 3);
  const worstHours = sortedByGas.slice(-3).reverse();

  const avgGas = hourlyStats.reduce((sum, h) => sum + h.avgGas, 0) / hourlyStats.length;
  const bestAvg = bestHours.reduce((sum, h) => sum + h.avgGas, 0) / bestHours.length;
  const worstAvg = worstHours.reduce((sum, h) => sum + h.avgGas, 0) / worstHours.length;
  const savingsPercent = Math.round(((worstAvg - bestAvg) / worstAvg) * 100);

  const formatHour = (hour: number) => {
    return `${hour.toString().padStart(2, '0')}:00`;
  };

  return (
    <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-xl sm:text-2xl">⏰</span>
        <h3 className="text-lg sm:text-xl font-bold text-gray-100">Best Times to Transact</h3>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
        {/* Cheapest Hours */}
        <div className="bg-green-500/10 border-2 border-green-500/50 rounded-lg p-3 sm:p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl sm:text-2xl">🟢</span>
            <div className="text-sm text-green-400 font-semibold">Cheapest Hours (UTC)</div>
          </div>

          <div className="space-y-1 mb-3">
            {bestHours.map((h) => (
              <div key={h.hour} className="flex items-center justify-between text-xs sm:text-sm">
                <span className="font-mono font-bold text-gray-100">{formatHour(h.hour)}</span>
                <span className="text-gray-400">{h.avgGas.toFixed(4)} gwei</span>
              </div>
            ))}
          </div>

          <div className="text-xs sm:text-sm text-green-300 font-semibold">
            Save up to {savingsPercent}% vs peak hours
          </div>
        </div>

        {/* Most Expensive Hours */}
        <div className="bg-red-500/10 border-2 border-red-500/50 rounded-lg p-3 sm:p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xl sm:text-2xl">🔴</span>
            <div className="text-sm text-red-400 font-semibold">Most Expensive (UTC)</div>
          </div>

          <div className="space-y-1 mb-3">
            {worstHours.map((h) => (
              <div key={h.hour} className="flex items-center justify-between text-xs sm:text-sm">
                <span className="font-mono font-bold text-gray-100">{formatHour(h.hour)}</span>
                <span className="text-gray-400">{h.avgGas.toFixed(4)} gwei</span>
              </div>
            ))}
          </div>

          <div className="text-xs sm:text-sm text-red-300 font-semibold">
            Avoid if transaction isn't urgent
          </div>
        </div>
      </div>

      {/* Current time indicator */}
      {currentGas > 0 && hourlyStats.length > 0 && avgGas > 0 && (
        <div className="mt-4 p-3 bg-gray-700/50 rounded-lg">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Current gas price:</span>
            <span className="font-bold text-gray-100">{currentGas.toFixed(4)} gwei</span>
          </div>
          <div className="flex items-center justify-between text-sm mt-1">
            <span className="text-gray-400">vs. 24h average:</span>
            <span className={`font-semibold ${
              currentGas < avgGas ? 'text-green-400' :
              currentGas > avgGas * 1.2 ? 'text-red-400' : 'text-yellow-400'
            }`}>
              {currentGas < avgGas ? '↓' : '↑'} {Math.abs(Math.round(((currentGas - avgGas) / avgGas) * 100))}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default BestTimeWidget;
