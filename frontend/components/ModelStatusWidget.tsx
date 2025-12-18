import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle, AlertTriangle, Clock, TrendingUp, Database, RefreshCw, Activity } from 'lucide-react';

interface ModelStatus {
  model_status: 'healthy' | 'warning' | 'needs_retraining';
  last_trained: string;
  model_version: string;
  performance_summary: {
    overall_accuracy: number;
    mae: number;
    r2_score: number;
  };
  next_check: string;
  days_since_training: number;
}

interface RetrainingHistory {
  id: number;
  timestamp: string;
  reason: string;
  models_trained: string[];
  validation_passed: boolean;
  performance_before: any;
  performance_after: any;
}

interface DataQuality {
  total_records: number;
  records_last_24h: number;
  records_last_7d: number;
  oldest_record: string;
  newest_record: string;
  has_sufficient_data: boolean;
  recommended_action: string;
}

const ModelStatusWidget: React.FC = () => {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [history, setHistory] = useState<RetrainingHistory[]>([]);
  const [dataQuality, setDataQuality] = useState<DataQuality | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [triggering, setTriggering] = useState(false);

  const API_BASE = import.meta.env.VITE_API_URL || 'https://basegasfeesml.onrender.com/api';

  useEffect(() => {
    fetchModelData();
    const interval = setInterval(fetchModelData, 120000); // Refresh every 2 minutes
    return () => clearInterval(interval);
  }, []);

  const fetchModelData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [statusRes, historyRes, dataRes] = await Promise.all([
        fetch(`${API_BASE}/retraining/status`),
        fetch(`${API_BASE}/retraining/history?limit=5`),
        fetch(`${API_BASE}/retraining/check-data`)
      ]);

      if (!statusRes.ok || !historyRes.ok || !dataRes.ok) {
        throw new Error('Failed to fetch model data');
      }

      const statusData = await statusRes.json();
      const historyData = await historyRes.json();
      const dataData = await dataRes.json();

      setStatus(statusData);
      setHistory(historyData.history || []);
      setDataQuality(dataData);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching model data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load model data');
      setLoading(false);
    }
  };

  const triggerRetraining = async () => {
    if (!confirm('Are you sure you want to trigger model retraining? This may take several minutes.')) {
      return;
    }

    try {
      setTriggering(true);
      const response = await fetch(`${API_BASE}/retraining/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: false })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to trigger retraining');
      }

      const result = await response.json();
      alert(result.message || 'Retraining completed successfully');
      fetchModelData(); // Refresh data
    } catch (err) {
      console.error('Error triggering retraining:', err);
      alert(err instanceof Error ? err.message : 'Failed to trigger retraining');
    } finally {
      setTriggering(false);
    }
  };

  const getStatusColor = (modelStatus: string) => {
    switch (modelStatus) {
      case 'healthy': return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'warning': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      case 'needs_retraining': return 'text-red-400 bg-red-500/20 border-red-500/30';
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  const getStatusIcon = (modelStatus: string) => {
    switch (modelStatus) {
      case 'healthy': return <CheckCircle className="w-5 h-5" />;
      case 'warning': return <AlertTriangle className="w-5 h-5" />;
      case 'needs_retraining': return <AlertTriangle className="w-5 h-5" />;
      default: return <Cpu className="w-5 h-5" />;
    }
  };

  const getStatusText = (modelStatus: string) => {
    switch (modelStatus) {
      case 'healthy': return 'Healthy';
      case 'warning': return 'Monitoring';
      case 'needs_retraining': return 'Needs Retraining';
      default: return 'Unknown';
    }
  };

  const formatPercentage = (value: number) => {
    if (value === undefined || value === null) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  if (loading && !status) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-4">
        <div className="flex items-center justify-center h-24">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-400"></div>
        </div>
      </div>
    );
  }

  if (error && !status) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-4">
        <div className="text-center text-red-400 text-sm">
          <AlertTriangle className="w-6 h-6 mx-auto mb-2" />
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl overflow-hidden">
      {/* Compact Header */}
      <div
        className="p-4 cursor-pointer hover:bg-slate-700/30 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-500/20 rounded-lg">
              <Cpu className="w-4 h-4 text-green-400" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">AI Model Status</h3>
              {status && (
                <p className="text-xs text-gray-400">
                  v{status.model_version} • Trained {status.days_since_training}d ago
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            {status && (
              <div className={`flex items-center gap-2 px-2.5 py-1 rounded-lg border ${getStatusColor(status.model_status)}`}>
                {getStatusIcon(status.model_status)}
                <span className="text-xs font-medium">{getStatusText(status.model_status)}</span>
              </div>
            )}
            <svg
              className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {isExpanded && status && (
        <div className="px-4 pb-4 space-y-4 border-t border-slate-700">
          {/* Performance Summary */}
          <div className="pt-4">
            <h4 className="text-xs font-medium text-gray-400 mb-3">Current Performance</h4>
            <div className="grid grid-cols-3 gap-3">
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <div className="flex items-center gap-1.5 mb-1">
                  <TrendingUp className="w-3 h-3 text-cyan-400" />
                  <p className="text-xs text-gray-400">Accuracy</p>
                </div>
                <p className="text-lg font-bold text-white">
                  {formatPercentage(status.performance_summary.overall_accuracy)}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <div className="flex items-center gap-1.5 mb-1">
                  <Activity className="w-3 h-3 text-blue-400" />
                  <p className="text-xs text-gray-400">MAE</p>
                </div>
                <p className="text-lg font-bold text-white">
                  {status.performance_summary.mae !== undefined && status.performance_summary.mae !== null ? status.performance_summary.mae.toFixed(6) : 'N/A'}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <div className="flex items-center gap-1.5 mb-1">
                  <CheckCircle className="w-3 h-3 text-green-400" />
                  <p className="text-xs text-gray-400">R² Score</p>
                </div>
                <p className="text-lg font-bold text-white">
                  {formatPercentage(status.performance_summary.r2_score)}
                </p>
              </div>
            </div>
          </div>

          {/* Data Quality */}
          {dataQuality && (
            <div>
              <h4 className="text-xs font-medium text-gray-400 mb-3">Training Data Quality</h4>
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <div>
                    <div className="flex items-center gap-1.5 mb-1">
                      <Database className="w-3 h-3 text-purple-400" />
                      <p className="text-xs text-gray-400">Total Records</p>
                    </div>
                    <p className="text-base font-bold text-white">
                      {dataQuality.total_records.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <div className="flex items-center gap-1.5 mb-1">
                      <Clock className="w-3 h-3 text-cyan-400" />
                      <p className="text-xs text-gray-400">Last 24h</p>
                    </div>
                    <p className="text-base font-bold text-white">
                      {dataQuality.records_last_24h.toLocaleString()}
                    </p>
                  </div>
                </div>
                {!dataQuality.has_sufficient_data && (
                  <div className="bg-yellow-500/10 border border-yellow-500/30 rounded px-2 py-1.5">
                    <p className="text-xs text-yellow-400">{dataQuality.recommended_action}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Training Info */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
              <div className="flex items-center gap-1.5 mb-1">
                <Clock className="w-3 h-3 text-gray-400" />
                <p className="text-xs text-gray-400">Last Trained</p>
              </div>
              <p className="text-xs font-medium text-white">
                {formatDate(status.last_trained)}
              </p>
            </div>
            <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
              <div className="flex items-center gap-1.5 mb-1">
                <RefreshCw className="w-3 h-3 text-gray-400" />
                <p className="text-xs text-gray-400">Next Check</p>
              </div>
              <p className="text-xs font-medium text-white">
                {formatDate(status.next_check)}
              </p>
            </div>
          </div>

          {/* Retraining History */}
          {history.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-gray-400 mb-2">Recent Retraining</h4>
              <div className="space-y-2">
                {history.slice(0, 3).map((item) => (
                  <div key={item.id} className="bg-slate-700/30 rounded-lg p-2.5 border border-slate-600">
                    <div className="flex items-start justify-between mb-1">
                      <div className="flex items-center gap-1.5">
                        {item.validation_passed ? (
                          <CheckCircle className="w-3 h-3 text-green-400" />
                        ) : (
                          <AlertTriangle className="w-3 h-3 text-red-400" />
                        )}
                        <span className="text-xs font-medium text-white">
                          {item.validation_passed ? 'Success' : 'Failed'}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">
                        {new Date(item.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-xs text-gray-400 mb-1">{item.reason}</p>
                    <p className="text-xs text-gray-500">
                      Models: {item.models_trained.join(', ')}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Manual Retrain Button */}
          <div className="pt-2 border-t border-slate-700">
            <button
              onClick={triggerRetraining}
              disabled={triggering}
              className="w-full px-3 py-2 bg-green-500/20 hover:bg-green-500/30 text-green-400 text-xs font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {triggering ? (
                <>
                  <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-green-400"></div>
                  Retraining...
                </>
              ) : (
                <>
                  <RefreshCw className="w-3 h-3" />
                  Trigger Manual Retraining
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelStatusWidget;
