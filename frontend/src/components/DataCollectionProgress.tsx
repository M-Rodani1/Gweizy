import React, { useState, useEffect, useRef } from 'react';
import { Database, TrendingUp, Calendar, Target, Zap } from 'lucide-react';
import { io, Socket } from 'socket.io-client';
import { API_CONFIG, getApiOrigin, getApiUrl } from '../config/api';

interface DataQuality {
  total_records: number;
  date_range_days: number;
  oldest_timestamp: string;
  newest_timestamp: string;
  recommended_days: number;
  sufficient_data: boolean;
  readiness: string;
  progress_percent: number;
}

const DataCollectionProgress: React.FC = () => {
  const [dataQuality, setDataQuality] = useState<DataQuality | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<Socket | null>(null);

  const SOCKET_URL = getApiOrigin();

  useEffect(() => {
    // Initial fetch
    fetchDataQuality();

    // Connect to WebSocket for real-time updates
    try {
      socketRef.current = io(SOCKET_URL, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 5
      });

      socketRef.current.on('connect', () => {
        console.log('✓ Connected to WebSocket for data quality updates');
      });

      socketRef.current.on('data_quality_update', (data: DataQuality) => {
        setDataQuality(data);
        setLoading(false);
        setError(null);
      });

      socketRef.current.on('connect_error', (err) => {
        console.warn('WebSocket connection error, falling back to polling:', err.message);
        // Fallback to polling if WebSocket fails
        const interval = setInterval(fetchDataQuality, 15000);
        return () => clearInterval(interval);
      });
    } catch (err) {
      console.warn('WebSocket not available, using polling fallback');
      const interval = setInterval(fetchDataQuality, 15000);
      return () => clearInterval(interval);
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  const fetchDataQuality = async () => {
    try {
      setError(null);
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.RETRAINING_CHECK_DATA, { t: Date.now() }), {
        cache: 'no-cache',
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch data quality');
      }

      const data = await response.json();
      setDataQuality(data);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching data quality:', err);
      setError(err instanceof Error ? err.message : 'Failed to load data');
      setLoading(false);
    }
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getEstimatedDaysRemaining = () => {
    if (!dataQuality) return 0;
    const daysNeeded = dataQuality.recommended_days - dataQuality.date_range_days;
    return Math.max(0, daysNeeded);
  };

  const getEstimatedCompletionDate = () => {
    if (!dataQuality) return '';
    const daysRemaining = getEstimatedDaysRemaining();
    const completionDate = new Date();
    completionDate.setDate(completionDate.getDate() + daysRemaining);
    return completionDate.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
  };

  const getRecordsPerDay = () => {
    if (!dataQuality || dataQuality.date_range_days === 0) return 0;
    return Math.round(dataQuality.total_records / dataQuality.date_range_days);
  };

  const getProgressColor = () => {
    if (!dataQuality) return 'bg-gray-500';
    if (dataQuality.progress_percent >= 100) return 'bg-green-500';
    if (dataQuality.progress_percent >= 50) return 'bg-blue-500';
    if (dataQuality.progress_percent >= 25) return 'bg-yellow-500';
    return 'bg-orange-500';
  };

  if (loading) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
        </div>
      </div>
    );
  }

  if (error || !dataQuality) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <div className="text-center text-red-400">
          <Database className="w-8 h-8 mx-auto mb-2" />
          <p>{error || 'Failed to load data collection status'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <Database className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Data Collection Progress</h3>
            <p className="text-xs text-gray-400">Training data collection status</p>
          </div>
        </div>
        <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
          dataQuality.sufficient_data
            ? 'bg-green-500/20 text-green-400 border border-green-500/30'
            : 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
        }`}>
          {dataQuality.readiness === 'ready' ? 'Ready' : 'Collecting'}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-2xl font-bold text-white">
              {Math.min(100, dataQuality.progress_percent).toFixed(1)}%
            </span>
            <span className="text-sm text-gray-400">Complete</span>
          </div>
          <div className="text-right">
            <div className="text-sm font-medium text-gray-300">
              {dataQuality.date_range_days} / {dataQuality.recommended_days} days
            </div>
            <div className="text-xs text-gray-500">
              {formatNumber(dataQuality.total_records)} records collected
            </div>
          </div>
        </div>

        {/* Enhanced Progress Bar with Segments */}
        <div className="relative pb-8">
          <div className="w-full bg-slate-700/50 rounded-full h-6 overflow-hidden shadow-inner border border-slate-600">
            <div
              className={`h-full ${getProgressColor()} transition-all duration-700 ease-out relative overflow-hidden`}
              style={{ width: `${Math.min(100, dataQuality.progress_percent)}%` }}
            >
              {/* Animated shimmer effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>

              {/* Progress segments */}
              <div className="absolute inset-0 flex">
                {[...Array(10)].map((_, i) => (
                  <div
                    key={i}
                    className="flex-1 border-r border-white/10 last:border-r-0"
                    style={{ opacity: (i + 1) * 10 <= dataQuality.progress_percent ? 1 : 0.3 }}
                  ></div>
                ))}
              </div>
            </div>
          </div>

          {/* Milestone markers */}
          <div className="absolute top-0 left-0 w-full h-6 pointer-events-none">
            {[0, 25, 50, 75, 100].map((milestone) => (
              <div
                key={milestone}
                className="absolute top-0 h-6"
                style={{
                  left: `${milestone}%`,
                  transform: milestone === 0 ? 'translateX(0)' : milestone === 100 ? 'translateX(-100%)' : 'translateX(-50%)'
                }}
              >
                <div className={`w-px h-full ${dataQuality.progress_percent >= milestone ? 'bg-white/30' : 'bg-slate-500/30'}`}></div>
              </div>
            ))}
          </div>

          {/* Milestone labels below the bar */}
          <div className="absolute top-8 left-0 w-full pointer-events-none">
            {[0, 25, 50, 75, 100].map((milestone) => (
              <div
                key={milestone}
                className={`absolute text-xs font-medium transition-colors whitespace-nowrap ${
                  dataQuality.progress_percent >= milestone ? 'text-cyan-400' : 'text-gray-600'
                }`}
                style={{
                  left: `${milestone}%`,
                  transform: milestone === 0 ? 'translateX(0)' : milestone === 100 ? 'translateX(-100%)' : 'translateX(-50%)'
                }}
              >
                {milestone}%
              </div>
            ))}
          </div>
        </div>

        {/* Days progress indicator */}
        <div className="mt-2 flex justify-between text-xs">
          <div className="text-gray-400">
            Started: {formatDate(dataQuality.oldest_timestamp).split(' ')[0]}
          </div>
          <div className="text-gray-400">
            Target: {dataQuality.recommended_days} days
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-1">
            <Database className="w-4 h-4 text-purple-400" />
            <p className="text-xs text-gray-400">Total Records</p>
          </div>
          <p className="text-2xl font-bold text-white">{formatNumber(dataQuality.total_records)}</p>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-4 h-4 text-yellow-400" />
            <p className="text-xs text-gray-400">Per Day</p>
          </div>
          <p className="text-2xl font-bold text-white">{formatNumber(getRecordsPerDay())}</p>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-1">
            <Calendar className="w-4 h-4 text-cyan-400" />
            <p className="text-xs text-gray-400">Days Collected</p>
          </div>
          <p className="text-2xl font-bold text-white">{dataQuality.date_range_days}</p>
        </div>

        <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
          <div className="flex items-center gap-2 mb-1">
            <Target className="w-4 h-4 text-green-400" />
            <p className="text-xs text-gray-400">Days Remaining</p>
          </div>
          <p className="text-2xl font-bold text-white">{getEstimatedDaysRemaining()}</p>
        </div>
      </div>

      {/* Timeline */}
      <div className="space-y-3 mb-6">
        <div className="flex items-center justify-between p-3 bg-slate-700/20 rounded-lg border border-slate-600">
          <div>
            <p className="text-xs text-gray-400 mb-1">Collection Started</p>
            <p className="text-sm font-medium text-white">{formatDate(dataQuality.oldest_timestamp)}</p>
          </div>
          <TrendingUp className="w-5 h-5 text-blue-400" />
        </div>

        <div className="flex items-center justify-between p-3 bg-slate-700/20 rounded-lg border border-slate-600">
          <div>
            <p className="text-xs text-gray-400 mb-1">Latest Data Point</p>
            <p className="text-sm font-medium text-white">{formatDate(dataQuality.newest_timestamp)}</p>
          </div>
          <Database className="w-5 h-5 text-green-400" />
        </div>

        {!dataQuality.sufficient_data && (
          <div className="flex items-center justify-between p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
            <div>
              <p className="text-xs text-blue-400 mb-1">Estimated Ready Date</p>
              <p className="text-sm font-medium text-white">{getEstimatedCompletionDate()}</p>
            </div>
            <Target className="w-5 h-5 text-blue-400" />
          </div>
        )}
      </div>

      {/* Collection Info */}
      <div className="bg-slate-700/20 rounded-lg p-4 border border-slate-600">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <Zap className="w-4 h-4 text-blue-400" />
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-white mb-1">Active Collection</p>
            <p className="text-xs text-gray-400 leading-relaxed">
              {dataQuality.sufficient_data
                ? 'Sufficient data collected! Ready for model training. Data collection continues for ongoing model improvements.'
                : `Collecting gas price data every 5 seconds. Once we reach ${dataQuality.recommended_days} days of historical data, we can train accurate ML models for price predictions.`
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataCollectionProgress;
