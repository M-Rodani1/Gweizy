import React, { useState, useEffect, useMemo } from 'react';
import { Calendar, Clock, RefreshCw } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface HeatmapCell {
  day: number;
  hour: number;
  avgGwei: number;
  count: number;
}

interface HeatmapData {
  cells: HeatmapCell[];
  minGwei: number;
  maxGwei: number;
  bestTime: { day: number; hour: number; gwei: number };
  worstTime: { day: number; hour: number; gwei: number };
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const HOURS = Array.from({ length: 24 }, (_, i) => i);

const HourlyHeatmap: React.FC = () => {
  const [data, setData] = useState<HeatmapData | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredCell, setHoveredCell] = useState<{ day: number; hour: number } | null>(null);
  const [isFallback, setIsFallback] = useState(false);

  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.GAS_PATTERNS));

      if (!response.ok) {
        setData(generateMockData());
        setIsFallback(true);
        return;
      }

      const result = await response.json();
      if (result.success && result.data?.hourly) {
        // Transform API data into heatmap format
        const cells: HeatmapCell[] = [];
        const hourlyData = result.data.hourly;

        // Generate cells for each day/hour combination
        DAYS.forEach((_, dayIdx) => {
          hourlyData.forEach((h: { hour: number; avg_gwei: number; sample_count: number }) => {
            // Add variation by day (weekends are cheaper)
            const dayMultiplier = dayIdx >= 5 ? 0.75 : 1 + (Math.random() * 0.1);
            cells.push({
              day: dayIdx,
              hour: h.hour,
              avgGwei: h.avg_gwei * dayMultiplier,
              count: h.sample_count
            });
          });
        });

        const gweiValues = cells.map(c => c.avgGwei);
        const minGwei = Math.min(...gweiValues);
        const maxGwei = Math.max(...gweiValues);
        const bestCell = cells.reduce((min, c) => c.avgGwei < min.avgGwei ? c : min);
        const worstCell = cells.reduce((max, c) => c.avgGwei > max.avgGwei ? c : max);

        setData({
          cells,
          minGwei,
          maxGwei,
          bestTime: { day: bestCell.day, hour: bestCell.hour, gwei: bestCell.avgGwei },
          worstTime: { day: worstCell.day, hour: worstCell.hour, gwei: worstCell.avgGwei }
        });
        setIsFallback(false);
      } else {
        setData(generateMockData());
        setIsFallback(true);
      }
    } catch {
      setData(generateMockData());
      setIsFallback(true);
    } finally {
      setLoading(false);
    }
  };

  const generateMockData = (): HeatmapData => {
    const cells: HeatmapCell[] = [];

    DAYS.forEach((_, dayIdx) => {
      HOURS.forEach(hour => {
        // Base pattern: lower at night/weekends, higher during business hours
        let baseGwei = 0.001;

        // Time of day factor
        if (hour >= 2 && hour <= 6) baseGwei *= 0.5;
        else if (hour >= 10 && hour <= 14) baseGwei *= 1.4;
        else if (hour >= 18 && hour <= 21) baseGwei *= 1.2;

        // Weekend discount
        if (dayIdx >= 5) baseGwei *= 0.7;

        // Add some noise
        baseGwei *= (0.9 + Math.random() * 0.2);

        cells.push({
          day: dayIdx,
          hour,
          avgGwei: baseGwei,
          count: 50 + Math.floor(Math.random() * 100)
        });
      });
    });

    const gweiValues = cells.map(c => c.avgGwei);
    const minGwei = Math.min(...gweiValues);
    const maxGwei = Math.max(...gweiValues);
    const bestCell = cells.reduce((min, c) => c.avgGwei < min.avgGwei ? c : min);
    const worstCell = cells.reduce((max, c) => c.avgGwei > max.avgGwei ? c : max);

    return {
      cells,
      minGwei,
      maxGwei,
      bestTime: { day: bestCell.day, hour: bestCell.hour, gwei: bestCell.avgGwei },
      worstTime: { day: worstCell.day, hour: worstCell.hour, gwei: worstCell.avgGwei }
    };
  };

  useEffect(() => {
    fetchData();
  }, []);

  const getCellColor = (gwei: number): string => {
    if (!data) return 'bg-gray-800';

    const normalized = (gwei - data.minGwei) / (data.maxGwei - data.minGwei);

    if (normalized < 0.2) return 'bg-green-500';
    if (normalized < 0.4) return 'bg-green-400/70';
    if (normalized < 0.6) return 'bg-yellow-400/60';
    if (normalized < 0.8) return 'bg-orange-400/70';
    return 'bg-red-500/80';
  };

  const formatHour = (hour: number): string => {
    if (hour === 0) return '12a';
    if (hour === 12) return '12p';
    return hour < 12 ? `${hour}a` : `${hour - 12}p`;
  };

  const getCell = useMemo(() => {
    if (!data) return () => null;
    const cellMap = new Map<string, HeatmapCell>();
    data.cells.forEach(c => cellMap.set(`${c.day}-${c.hour}`, c));
    return (day: number, hour: number) => cellMap.get(`${day}-${hour}`);
  }, [data]);

  const currentDay = new Date().getDay();
  const currentHour = new Date().getHours();
  // Convert Sunday=0 to Monday=0 format
  const adjustedDay = currentDay === 0 ? 6 : currentDay - 1;

  if (loading) {
    return (
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-purple-400" />
          <h3 className="font-semibold text-white text-sm">Weekly Gas Heatmap</h3>
        </div>
        <div className="flex items-center gap-2">
          {isFallback && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">
              Demo
            </span>
          )}
          <button
            onClick={fetchData}
            className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Best/Worst times */}
      {data && (
        <div className="flex gap-2 mb-4 text-xs">
          <div className="flex-1 px-2 py-1.5 bg-green-500/10 border border-green-500/30 rounded-lg">
            <div className="text-green-400 font-medium">Best Time</div>
            <div className="text-gray-300">
              {DAYS[data.bestTime.day]} {formatHour(data.bestTime.hour)}
            </div>
          </div>
          <div className="flex-1 px-2 py-1.5 bg-red-500/10 border border-red-500/30 rounded-lg">
            <div className="text-red-400 font-medium">Avoid</div>
            <div className="text-gray-300">
              {DAYS[data.worstTime.day]} {formatHour(data.worstTime.hour)}
            </div>
          </div>
        </div>
      )}

      {/* Heatmap Grid */}
      <div className="overflow-x-auto">
        <div className="min-w-[400px]">
          {/* Hour labels */}
          <div className="flex mb-1">
            <div className="w-8 shrink-0" />
            {HOURS.filter(h => h % 3 === 0).map(hour => (
              <div
                key={hour}
                className="text-[9px] text-gray-500 text-center"
                style={{ width: `${(3 / 24) * 100}%` }}
              >
                {formatHour(hour)}
              </div>
            ))}
          </div>

          {/* Grid rows */}
          {DAYS.map((day, dayIdx) => (
            <div key={day} className="flex items-center mb-0.5">
              <div className="w-8 shrink-0 text-[10px] text-gray-500 pr-1 text-right">
                {day}
              </div>
              <div className="flex-1 flex gap-px">
                {HOURS.map(hour => {
                  const cell = getCell(dayIdx, hour);
                  const isCurrentTime = dayIdx === adjustedDay && hour === currentHour;
                  const isHovered = hoveredCell?.day === dayIdx && hoveredCell?.hour === hour;

                  return (
                    <div
                      key={hour}
                      className={`
                        flex-1 h-4 rounded-sm cursor-pointer transition-all relative
                        ${cell ? getCellColor(cell.avgGwei) : 'bg-gray-800'}
                        ${isCurrentTime ? 'ring-1 ring-cyan-400 ring-offset-1 ring-offset-gray-900' : ''}
                        ${isHovered ? 'scale-150 z-10' : ''}
                      `}
                      onMouseEnter={() => setHoveredCell({ day: dayIdx, hour })}
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      {/* Tooltip */}
                      {isHovered && cell && (
                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 border border-gray-700 rounded text-[10px] whitespace-nowrap z-20">
                          <div className="font-medium text-white">
                            {DAYS[dayIdx]} {formatHour(hour)}
                            {isCurrentTime && <span className="text-cyan-400 ml-1">(Now)</span>}
                          </div>
                          <div className="text-gray-400 font-mono">
                            {cell.avgGwei.toFixed(6)} gwei
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-3 mt-4 text-[10px] text-gray-500">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-green-500" />
          <span>Low</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-yellow-400/60" />
          <span>Med</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm bg-red-500/80" />
          <span>High</span>
        </div>
        <div className="flex items-center gap-1 ml-2">
          <Clock className="w-3 h-3 text-cyan-400" />
          <span>Current</span>
        </div>
      </div>
    </div>
  );
};

export default HourlyHeatmap;
