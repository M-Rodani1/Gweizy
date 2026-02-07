import React, { useState, useEffect } from 'react';
import { Calendar, Clock, TrendingDown, TrendingUp, Loader2 } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface HourlyPattern {
  hour: number;
  avg_gwei: number;
  min_gwei: number;
  max_gwei: number;
  sample_count: number;
}

interface DailyPattern {
  day: number; // 0 = Monday, 6 = Sunday
  avg_gwei: number;
  min_gwei: number;
  max_gwei: number;
}

interface HeatmapData {
  hourly: HourlyPattern[];
  daily: DailyPattern[];
  overall_avg: number;
  cheapest_hour: number;
  most_expensive_hour: number;
  cheapest_day: number;
  most_expensive_day: number;
}

const GasPatternHeatmap: React.FC = () => {
  const [data, setData] = useState<HeatmapData | null>(null);
  const [loading, setLoading] = useState(true);
  const [_error, _setError] = useState<string | null>(null);
  const [view, setView] = useState<'hourly' | 'daily'>('hourly');
  const [isFallback, setIsFallback] = useState(false);

  const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const HOURS = Array.from({ length: 24 }, (_, i) => i);

  useEffect(() => {
    fetchPatternData();
  }, []);

  const fetchPatternData = async () => {
    try {
      setLoading(true);
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.GAS_PATTERNS));

      if (!response.ok) {
        // Generate mock data if API not available
        setData(generateMockData());
        setIsFallback(true);
        setLoading(false);
        return;
      }

      const result = await response.json();
      if (result.success) {
        setData(result.data);
        setIsFallback(false);
      } else {
        setData(generateMockData());
        setIsFallback(true);
      }
      setLoading(false);
    } catch (err) {
      // Use mock data for demo
      setData(generateMockData());
      setIsFallback(true);
      setLoading(false);
    }
  };

  const generateMockData = (): HeatmapData => {
    // Generate realistic-looking pattern data
    const hourly: HourlyPattern[] = HOURS.map(hour => {
      // Gas tends to be lower at night (2-6 AM) and higher during peak hours (10 AM - 2 PM)
      const baseGwei = 0.001;
      let multiplier = 1;

      if (hour >= 2 && hour <= 6) {
        multiplier = 0.6 + Math.random() * 0.2; // Lowest at night
      } else if (hour >= 10 && hour <= 14) {
        multiplier = 1.3 + Math.random() * 0.3; // Highest midday
      } else if (hour >= 18 && hour <= 21) {
        multiplier = 1.1 + Math.random() * 0.2; // Slightly higher evening
      } else {
        multiplier = 0.9 + Math.random() * 0.2;
      }

      const avg = baseGwei * multiplier;
      return {
        hour,
        avg_gwei: avg,
        min_gwei: avg * 0.7,
        max_gwei: avg * 1.4,
        sample_count: 100 + Math.floor(Math.random() * 50)
      };
    });

    const daily: DailyPattern[] = DAYS.map((_, day) => {
      const baseGwei = 0.001;
      // Weekends tend to have lower gas
      const multiplier = day >= 5 ? 0.7 + Math.random() * 0.2 : 0.9 + Math.random() * 0.3;
      const avg = baseGwei * multiplier;

      return {
        day,
        avg_gwei: avg,
        min_gwei: avg * 0.6,
        max_gwei: avg * 1.5
      };
    });

    const cheapestHour = hourly.reduce((min, h) => h.avg_gwei < min.avg_gwei ? h : min).hour;
    const mostExpensiveHour = hourly.reduce((max, h) => h.avg_gwei > max.avg_gwei ? h : max).hour;
    const cheapestDay = daily.reduce((min, d) => d.avg_gwei < min.avg_gwei ? d : min).day;
    const mostExpensiveDay = daily.reduce((max, d) => d.avg_gwei > max.avg_gwei ? d : max).day;

    return {
      hourly,
      daily,
      overall_avg: 0.001,
      cheapest_hour: cheapestHour,
      most_expensive_hour: mostExpensiveHour,
      cheapest_day: cheapestDay,
      most_expensive_day: mostExpensiveDay
    };
  };

  const getHeatColor = (value: number, min: number, max: number): string => {
    const normalized = (value - min) / (max - min);

    if (normalized < 0.25) {
      return 'bg-green-500/80';
    } else if (normalized < 0.5) {
      return 'bg-green-400/60';
    } else if (normalized < 0.75) {
      return 'bg-yellow-400/60';
    } else {
      return 'bg-red-400/70';
    }
  };

  const formatHour = (hour: number): string => {
    if (hour === 0) return '12 AM';
    if (hour === 12) return '12 PM';
    return hour < 12 ? `${hour} AM` : `${hour - 12} PM`;
  };

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl">
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl">
        <div className="text-center py-8 text-gray-400">
          <Calendar className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>Not enough data to show patterns yet</p>
          <p className="text-xs mt-2">Check back after a few days of data collection</p>
        </div>
      </div>
    );
  }

  const hourlyMin = Math.min(...data.hourly.map(h => h.avg_gwei));
  const hourlyMax = Math.max(...data.hourly.map(h => h.avg_gwei));
  const dailyMin = Math.min(...data.daily.map(d => d.avg_gwei));
  const dailyMax = Math.max(...data.daily.map(d => d.avg_gwei));

  return (
    <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Calendar className="w-5 h-5 text-purple-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Gas Price Patterns</h3>
            <p className="text-xs text-gray-400">Historical averages to find the best time to transact</p>
          </div>
        </div>
        {isFallback && (
          <span className="text-xs px-2 py-1 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-300">
            Using simulated data
          </span>
        )}

        {/* View Toggle */}
        <div className="flex bg-slate-700/50 rounded-lg p-1">
          <button
            onClick={() => setView('hourly')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              view === 'hourly'
                ? 'bg-purple-500 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Clock className="w-4 h-4 inline mr-1" />
            By Hour
          </button>
          <button
            onClick={() => setView('daily')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              view === 'daily'
                ? 'bg-purple-500 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Calendar className="w-4 h-4 inline mr-1" />
            By Day
          </button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <TrendingDown className="w-4 h-4 text-green-400" />
            <span className="text-xs text-gray-400">Cheapest Time</span>
          </div>
          <p className="text-lg font-bold text-green-400">
            {view === 'hourly'
              ? formatHour(data.cheapest_hour)
              : DAYS[data.cheapest_day]}
          </p>
        </div>
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="w-4 h-4 text-red-400" />
            <span className="text-xs text-gray-400">Most Expensive</span>
          </div>
          <p className="text-lg font-bold text-red-400">
            {view === 'hourly'
              ? formatHour(data.most_expensive_hour)
              : DAYS[data.most_expensive_day]}
          </p>
        </div>
      </div>

      {/* Heatmap */}
      {view === 'hourly' ? (
        <div className="space-y-1">
          {/* Time labels - top row */}
          <div className="grid grid-cols-12 gap-1 mb-2">
            {[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22].map((hour) => (
              <div key={hour} className="text-xs text-gray-500 text-center">
                {formatHour(hour).replace(' AM', 'a').replace(' PM', 'p')}
              </div>
            ))}
          </div>

          {/* Heatmap grid */}
          <div className="grid grid-cols-24 gap-0.5">
            {data.hourly.map((hour, index) => {
              const currentHour = new Date().getHours();
              const isCurrentHour = hour.hour === currentHour;

              return (
                <div
                  key={hour.hour}
                  className={`
                    h-10 rounded-sm ${getHeatColor(hour.avg_gwei, hourlyMin, hourlyMax)}
                    cursor-pointer transition-transform hover:scale-110 relative group
                    heatmap-cell ${isCurrentHour ? 'current-hour-marker ring-2 ring-cyan-400 ring-offset-1 ring-offset-slate-800' : ''}
                  `}
                  style={{ animationDelay: `${index * 30}ms` }}
                  title={`${formatHour(hour.hour)}: ${hour.avg_gwei.toFixed(6)} gwei avg`}
                >
                  {/* Current hour indicator */}
                  {isCurrentHour && (
                    <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                  )}
                  {/* Tooltip */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 border border-slate-600 rounded text-xs text-white whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-10 pointer-events-none">
                    <div className="font-medium flex items-center gap-1">
                      {formatHour(hour.hour)}
                      {isCurrentHour && <span className="text-cyan-400">(Now)</span>}
                    </div>
                    <div className="text-gray-400 font-mono">Avg: {hour.avg_gwei.toFixed(6)} gwei</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div className="flex items-center justify-center gap-4 mt-4 text-xs text-gray-400">
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-green-500/80"></div>
              <span>Low</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-yellow-400/60"></div>
              <span>Medium</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-4 h-4 rounded bg-red-400/70"></div>
              <span>High</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {data.daily.map((day) => (
            <div key={day.day} className="flex items-center gap-3">
              <div className="w-12 text-sm text-gray-400">{DAYS[day.day]}</div>
              <div className="flex-1 h-8 relative rounded overflow-hidden bg-slate-700/30">
                <div
                  className={`absolute inset-y-0 left-0 ${getHeatColor(day.avg_gwei, dailyMin, dailyMax)} transition-all`}
                  style={{ width: `${((day.avg_gwei - dailyMin) / (dailyMax - dailyMin)) * 100 + 20}%` }}
                />
                <div className="absolute inset-0 flex items-center px-3">
                  <span className="text-sm font-medium text-white">
                    {day.avg_gwei.toFixed(6)} gwei
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Insight */}
      <div className="mt-6 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
        <p className="text-sm text-purple-300">
          <strong>Tip:</strong> {view === 'hourly'
            ? `Gas prices are typically lowest around ${formatHour(data.cheapest_hour)} and highest around ${formatHour(data.most_expensive_hour)}.`
            : `${DAYS[data.cheapest_day]} tends to have the lowest gas prices, while ${DAYS[data.most_expensive_day]} is usually the most expensive.`
          }
        </p>
      </div>
    </div>
  );
};

export default GasPatternHeatmap;
