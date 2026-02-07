import React, { useEffect, useState, useMemo } from 'react';
import { TrendingUp, RefreshCw, Calendar, ChevronDown } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface HistoryPoint {
  timestamp: string;
  mae: number;
  rmse: number;
  r2: number;
  directional_accuracy: number;
  n: number;
}

interface HistoryData {
  '1h': HistoryPoint[];
  '4h': HistoryPoint[];
  '24h': HistoryPoint[];
}

type MetricType = 'mae' | 'r2' | 'directional_accuracy';
type TimeRange = '24h' | '7d' | '30d';

const AccuracyHistoryChart: React.FC = () => {
  const [history, setHistory] = useState<HistoryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedHorizon, setSelectedHorizon] = useState<'1h' | '4h' | '24h'>('1h');
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('mae');
  const [timeRange, setTimeRange] = useState<TimeRange>('7d');

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const hoursBack = timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 720;
      const resolution = timeRange === '30d' ? 'daily' : 'hourly';

      const response = await fetch(
        getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_HISTORY) + `?hours_back=${hoursBack}&resolution=${resolution}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch history');
      }

      const data = await response.json();
      if (data.success && data.history) {
        setHistory(data.history);
        setError(null);
      } else {
        throw new Error('No history data');
      }
    } catch (err) {
      setError('Could not load history');
      // Generate mock data for demo
      const mockHistory: HistoryData = { '1h': [], '4h': [], '24h': [] };
      const now = new Date();
      const points = timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 30;

      for (let i = points; i >= 0; i--) {
        const date = new Date(now.getTime() - i * (timeRange === '30d' ? 86400000 : 3600000));
        const timestamp = timeRange === '30d'
          ? date.toISOString().split('T')[0]
          : date.toISOString().slice(0, 13) + ':00';

        ['1h', '4h', '24h'].forEach(horizon => {
          const baseMAE = horizon === '1h' ? 0.0003 : horizon === '4h' ? 0.0004 : 0.0006;
          mockHistory[horizon as keyof HistoryData].push({
            timestamp,
            mae: baseMAE * (0.8 + Math.random() * 0.4),
            rmse: baseMAE * 1.3 * (0.8 + Math.random() * 0.4),
            r2: 0.5 + Math.random() * 0.4,
            directional_accuracy: 0.5 + Math.random() * 0.2,
            n: Math.floor(10 + Math.random() * 20)
          });
        });
      }
      setHistory(mockHistory);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [timeRange]);

  const getMetricValue = (point: HistoryPoint): number => {
    switch (selectedMetric) {
      case 'mae': return point.mae;
      case 'r2': return point.r2;
      case 'directional_accuracy': return point.directional_accuracy;
      default: return 0;
    }
  };

  const formatMetricValue = (value: number): string => {
    if (selectedMetric === 'mae') {
      return value < 0.001 ? value.toExponential(2) : value.toFixed(6);
    }
    return (value * 100).toFixed(1) + '%';
  };

  const getMetricLabel = (): string => {
    switch (selectedMetric) {
      case 'mae': return 'MAE (Mean Absolute Error)';
      case 'r2': return 'R² Score';
      case 'directional_accuracy': return 'Directional Accuracy';
      default: return '';
    }
  };

  const getMetricColor = (): string => {
    switch (selectedMetric) {
      case 'mae': return '#f59e0b'; // amber
      case 'r2': return '#06b6d4'; // cyan
      case 'directional_accuracy': return '#06b6d4'; // cyan
      default: return '#6b7280';
    }
  };

  const currentData = history?.[selectedHorizon] || [];
  const maxValue = Math.max(...currentData.map(getMetricValue), 0.001);
  const minValue = Math.min(...currentData.map(getMetricValue), 0);

  // Calculate trend
  const calcTrend = (): { direction: 'up' | 'down' | 'flat'; percent: number } => {
    if (currentData.length < 2) return { direction: 'flat', percent: 0 };
    const recent = currentData.slice(-5);
    const earlier = currentData.slice(0, 5);
    if (recent.length === 0 || earlier.length === 0) return { direction: 'flat', percent: 0 };

    const recentAvg = recent.reduce((a, b) => a + getMetricValue(b), 0) / recent.length;
    const earlierAvg = earlier.reduce((a, b) => a + getMetricValue(b), 0) / earlier.length;
    const change = ((recentAvg - earlierAvg) / earlierAvg) * 100;

    return {
      direction: Math.abs(change) < 5 ? 'flat' : change > 0 ? 'up' : 'down',
      percent: Math.abs(change)
    };
  };

  const trend = calcTrend();

  // Generate screen reader summary
  const chartSummary = useMemo(() => {
    if (currentData.length === 0) return 'No historical accuracy data available.';

    const latestValue = getMetricValue(currentData[currentData.length - 1]);
    const metricName = getMetricLabel();
    const trendDesc = trend.direction === 'flat' ? 'stable' :
      trend.direction === 'up' ? 'increasing' : 'decreasing';

    return `Accuracy history for ${selectedHorizon} predictions over ${timeRange}. ` +
      `Current ${metricName}: ${formatMetricValue(latestValue)}. ` +
      `Trend is ${trendDesc} by ${trend.percent.toFixed(1)}%. ` +
      `Based on ${currentData.reduce((a, b) => a + b.n, 0)} data points.`;
  }, [currentData, selectedHorizon, timeRange, trend, selectedMetric]);

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-cyan-400" aria-hidden="true" />
          <h3 className="font-semibold text-white">Accuracy History</h3>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Time Range Selector */}
          <div className="relative">
            <label htmlFor="time-range-select" className="sr-only">Select time range</label>
            <select
              id="time-range-select"
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as TimeRange)}
              className="appearance-none bg-gray-800 text-gray-300 text-xs px-3 py-1.5 pr-7 rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
            >
              <option value="24h">24 Hours</option>
              <option value="7d">7 Days</option>
              <option value="30d">30 Days</option>
            </select>
            <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-gray-500 pointer-events-none" aria-hidden="true" />
          </div>

          {/* Refresh */}
          <button
            onClick={fetchHistory}
            aria-label="Refresh accuracy history"
            className="text-xs text-cyan-300 hover:text-cyan-200 transition-colors flex items-center gap-1 px-2 py-1.5 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500"
            disabled={loading}
          >
            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Screen reader description */}
      <p className="sr-only">{chartSummary}</p>

      {/* Controls */}
      <div className="flex flex-wrap gap-2 mb-4">
        {/* Horizon Tabs */}
        <div className="flex gap-1 bg-gray-800/50 rounded-lg p-1" role="group" aria-label="Select prediction horizon">
          {(['1h', '4h', '24h'] as const).map((horizon) => (
            <button
              key={horizon}
              onClick={() => setSelectedHorizon(horizon)}
              aria-pressed={selectedHorizon === horizon}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
                selectedHorizon === horizon
                  ? 'bg-cyan-500 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {horizon}
            </button>
          ))}
        </div>

        {/* Metric Selector */}
        <div className="flex gap-1 bg-gray-800/50 rounded-lg p-1" role="group" aria-label="Select metric type">
          {([
            { key: 'mae', label: 'MAE' },
            { key: 'r2', label: 'R²' },
            { key: 'directional_accuracy', label: 'Dir.' }
          ] as const).map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setSelectedMetric(key)}
              aria-pressed={selectedMetric === key}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500 ${
                selectedMetric === key
                  ? 'bg-cyan-500 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      {loading && !history ? (
        <div className="flex items-center justify-center py-12" role="status" aria-label="Loading">
          <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" aria-hidden="true" />
          <span className="sr-only">Loading accuracy history...</span>
        </div>
      ) : currentData.length === 0 ? (
        <div className="flex items-center justify-center py-12 text-gray-500 text-sm">
          No historical data available
        </div>
      ) : (
        <>
          {/* Trend Summary */}
          <div className="flex items-center gap-4 mb-4 p-3 bg-gray-800/40 rounded-xl">
            <div className="flex-1">
              <div className="text-xs text-gray-500 mb-1">{getMetricLabel()}</div>
              <div className="text-lg font-mono font-bold" style={{ color: getMetricColor() }}>
                {formatMetricValue(getMetricValue(currentData[currentData.length - 1]))}
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500 mb-1">Trend</div>
              <div className={`text-sm font-semibold flex items-center gap-1 ${
                trend.direction === 'up'
                  ? selectedMetric === 'mae' ? 'text-red-400' : 'text-green-400'
                  : trend.direction === 'down'
                    ? selectedMetric === 'mae' ? 'text-green-400' : 'text-red-400'
                    : 'text-gray-400'
              }`}>
                {trend.direction === 'up' ? '↑' : trend.direction === 'down' ? '↓' : '→'}
                {trend.percent.toFixed(1)}%
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500 mb-1">Data Points</div>
              <div className="text-sm font-mono text-gray-300">
                {currentData.reduce((a, b) => a + b.n, 0)}
              </div>
            </div>
          </div>

          {/* Sparkline Chart */}
          <div className="relative h-32 sm:h-40" role="img" aria-label={`${getMetricLabel()} chart for ${selectedHorizon} predictions`}>
            <svg className="w-full h-full" viewBox={`0 0 ${currentData.length * 10} 100`} preserveAspectRatio="none" aria-hidden="true">
              {/* Grid lines */}
              {[0, 25, 50, 75, 100].map((y) => (
                <line
                  key={y}
                  x1="0"
                  y1={y}
                  x2={currentData.length * 10}
                  y2={y}
                  stroke="#374151"
                  strokeWidth="0.5"
                />
              ))}

              {/* Data line */}
              <polyline
                fill="none"
                stroke={getMetricColor()}
                strokeWidth="2"
                points={currentData.map((point, i) => {
                  const value = getMetricValue(point);
                  const y = 100 - ((value - minValue) / (maxValue - minValue)) * 90 - 5;
                  return `${i * 10 + 5},${y}`;
                }).join(' ')}
              />

              {/* Area fill */}
              <polygon
                fill={`${getMetricColor()}20`}
                points={`0,100 ${currentData.map((point, i) => {
                  const value = getMetricValue(point);
                  const y = 100 - ((value - minValue) / (maxValue - minValue)) * 90 - 5;
                  return `${i * 10 + 5},${y}`;
                }).join(' ')} ${currentData.length * 10},100`}
              />

              {/* Data points */}
              {currentData.map((point, i) => {
                const value = getMetricValue(point);
                const y = 100 - ((value - minValue) / (maxValue - minValue)) * 90 - 5;
                return (
                  <circle
                    key={i}
                    cx={i * 10 + 5}
                    cy={y}
                    r="2"
                    fill={getMetricColor()}
                    className="opacity-0 hover:opacity-100 transition-opacity"
                  />
                );
              })}
            </svg>

            {/* Y-axis labels */}
            <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-gray-500 font-mono">
              <span>{formatMetricValue(maxValue)}</span>
              <span>{formatMetricValue(minValue)}</span>
            </div>
          </div>

          {/* X-axis labels */}
          <div className="flex justify-between text-xs text-gray-500 mt-2 px-4">
            <span>{currentData[0]?.timestamp?.split(' ')[0] || ''}</span>
            <span>{currentData[currentData.length - 1]?.timestamp?.split(' ')[0] || ''}</span>
          </div>
        </>
      )}

      {/* Footer */}
      {error && (
        <div className="mt-3 text-xs text-amber-400/80 flex items-center gap-1" role="status">
          <Calendar className="w-3 h-3" aria-hidden="true" />
          Using demo data
        </div>
      )}
    </div>
  );
};

export default AccuracyHistoryChart;
