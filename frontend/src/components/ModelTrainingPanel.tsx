/**
 * Model Training Panel Component
 * Displays ML model status and allows triggering model training
 */

import React, { useState, useEffect, useCallback, memo } from 'react';
import {
  Brain,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Cpu,
  Database,
  Zap,
  Play
} from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface ModelStatus {
  available: boolean;
  path: string | null;
}

interface ModelsStatusResponse {
  prediction_models: Record<string, ModelStatus>;
  spike_detectors: Record<string, ModelStatus>;
  dqn_agent: ModelStatus;
  data_status: {
    total_records: number;
    sufficient_for_training: boolean;
    sufficient_for_dqn: boolean;
    error?: string;
  };
  overall_ready: boolean;
  missing_models: string[];
  summary: {
    prediction_models_ready: boolean;
    spike_detectors_ready: boolean;
    dqn_agent_ready: boolean;
    action_needed: string | null;
  };
}

interface TrainingResponse {
  status: 'started' | 'in_progress' | 'error';
  message: string;
  steps?: string[];
  timestamp?: string;
  note?: string;
}

const ModelTrainingPanel: React.FC = () => {
  const [status, setStatus] = useState<ModelsStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [training, setTraining] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      setError(null);
      const response = await fetch(getApiUrl('/retraining/models-status'));

      if (!response.ok) {
        throw new Error('Failed to fetch model status');
      }

      const data: ModelsStatusResponse = await response.json();
      setStatus(data);
      setLastUpdated(new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const triggerTraining = async () => {
    if (!confirm('Start model training? This may take 5-15 minutes and runs in the background.')) {
      return;
    }

    setTraining(true);
    setTrainingMessage(null);

    try {
      const response = await fetch(getApiUrl('/retraining/simple'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const data: TrainingResponse = await response.json();

      if (data.status === 'started' || data.status === 'in_progress') {
        setTrainingMessage(data.message);
        // Poll for status updates
        setTimeout(fetchStatus, 5000);
      } else {
        setError(data.message || 'Failed to start training');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger training');
    } finally {
      setTraining(false);
    }
  };

  const getStatusIcon = (available: boolean) => {
    return available ? (
      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
    ) : (
      <XCircle className="w-4 h-4 text-red-400" />
    );
  };

  const getStatusBadge = (ready: boolean, label: string) => {
    return (
      <span className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${
        ready
          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
          : 'bg-red-500/20 text-red-400 border border-red-500/30'
      }`}>
        {ready ? <CheckCircle2 className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
        {label}
      </span>
    );
  };

  if (loading && !status) {
    return (
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl h-full">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">ML Model Training</h3>
        </div>
        <div className="flex items-center justify-center h-40">
          <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
        </div>
      </div>
    );
  }

  if (error && !status) {
    return (
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl h-full">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">ML Model Training</h3>
        </div>
        <div className="text-center py-8">
          <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-3" />
          <p className="text-red-400 text-sm">{error}</p>
          <button
            onClick={fetchStatus}
            className="mt-4 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white text-sm rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">ML Model Training</h3>
        </div>
        <button
          onClick={fetchStatus}
          disabled={loading}
          className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-1"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Overall Status */}
      {status && (
        <>
          <div className={`rounded-xl p-4 mb-4 border ${
            status.overall_ready
              ? 'bg-emerald-500/10 border-emerald-500/30'
              : 'bg-amber-500/10 border-amber-500/30'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {status.overall_ready ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                )}
                <span className={`font-medium ${
                  status.overall_ready ? 'text-emerald-400' : 'text-amber-400'
                }`}>
                  {status.overall_ready ? 'All Models Ready' : 'Models Need Training'}
                </span>
              </div>
              {!status.overall_ready && (
                <span className="text-xs text-gray-400">
                  {status.missing_models.length} missing
                </span>
              )}
            </div>
          </div>

          {/* Model Categories */}
          <div className="space-y-3 flex-1">
            {/* Prediction Models */}
            <div className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm text-gray-300">Prediction Models</span>
                </div>
                {getStatusBadge(status.summary.prediction_models_ready,
                  status.summary.prediction_models_ready ? 'Ready' : 'Missing')}
              </div>
              <div className="flex gap-2">
                {['1h', '4h', '24h'].map(horizon => (
                  <div
                    key={horizon}
                    className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
                      status.prediction_models[horizon]?.available
                        ? 'bg-emerald-500/10 text-emerald-400'
                        : 'bg-red-500/10 text-red-400'
                    }`}
                  >
                    {getStatusIcon(status.prediction_models[horizon]?.available)}
                    {horizon}
                  </div>
                ))}
              </div>
            </div>

            {/* Spike Detectors */}
            <div className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  <span className="text-sm text-gray-300">Spike Detectors</span>
                </div>
                {getStatusBadge(status.summary.spike_detectors_ready,
                  status.summary.spike_detectors_ready ? 'Ready' : 'Missing')}
              </div>
              <div className="flex gap-2">
                {['1h', '4h', '24h'].map(horizon => (
                  <div
                    key={horizon}
                    className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
                      status.spike_detectors[horizon]?.available
                        ? 'bg-emerald-500/10 text-emerald-400'
                        : 'bg-red-500/10 text-red-400'
                    }`}
                  >
                    {getStatusIcon(status.spike_detectors[horizon]?.available)}
                    {horizon}
                  </div>
                ))}
              </div>
            </div>

            {/* DQN Agent */}
            <div className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-purple-400" />
                  <span className="text-sm text-gray-300">DQN Agent</span>
                </div>
                {getStatusBadge(status.summary.dqn_agent_ready,
                  status.summary.dqn_agent_ready ? 'Ready' : 'Missing')}
              </div>
            </div>

            {/* Data Status */}
            <div className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-gray-300">Training Data</span>
                </div>
                <span className="text-xs text-gray-400">
                  {status.data_status.total_records?.toLocaleString() || 0} records
                </span>
              </div>
              {status.data_status.error ? (
                <p className="text-xs text-red-400 mt-1">{status.data_status.error}</p>
              ) : (
                <div className="flex gap-2 mt-2">
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    status.data_status.sufficient_for_training
                      ? 'bg-emerald-500/10 text-emerald-400'
                      : 'bg-red-500/10 text-red-400'
                  }`}>
                    Basic: {status.data_status.sufficient_for_training ? 'OK' : 'Need 50+'}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    status.data_status.sufficient_for_dqn
                      ? 'bg-emerald-500/10 text-emerald-400'
                      : 'bg-amber-500/10 text-amber-400'
                  }`}>
                    DQN: {status.data_status.sufficient_for_dqn ? 'OK' : 'Need 500+'}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Training Message */}
          {trainingMessage && (
            <div className="mt-4 p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-xl">
              <p className="text-sm text-cyan-400">{trainingMessage}</p>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {/* Train Button */}
          {!status.overall_ready && (
            <button
              onClick={triggerTraining}
              disabled={training || !status.data_status.sufficient_for_training}
              className="mt-4 w-full px-4 py-3 bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-400 hover:to-cyan-400 text-white font-medium rounded-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all shadow-lg shadow-purple-500/20"
            >
              {training ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Starting Training...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Train All Models
                </>
              )}
            </button>
          )}

          {!status.data_status.sufficient_for_training && (
            <p className="mt-2 text-xs text-center text-amber-400">
              Need at least 50 data records to train models
            </p>
          )}
        </>
      )}

      {/* Footer */}
      <div className="mt-auto pt-4 border-t border-gray-800/50">
        <div className="text-xs text-gray-500">
          Last check: {lastUpdated || '...'}
        </div>
      </div>
    </div>
  );
};

export default memo(ModelTrainingPanel);
