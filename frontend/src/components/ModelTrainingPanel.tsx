/**
 * Model Training Panel Component
 * Displays ML model status and allows triggering model training
 * Shows real-time training progress with step indicators
 */

import React, { useState, useEffect, useCallback, memo, useRef } from 'react';
import {
  Brain,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Cpu,
  Database,
  Zap,
  Play,
  Loader2,
  Clock,
  SkipForward,
  History,
  ChevronDown,
  ChevronUp,
  FileText
} from 'lucide-react';
import { getApiUrl } from '../config/api';

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

interface TrainingStep {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  message: string | null;
}

interface TrainingProgress {
  is_training: boolean;
  current_step: number;
  total_steps: number;
  step_name: string | null;
  step_status: string | null;
  steps: TrainingStep[];
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

interface TrainingResponse {
  status: 'started' | 'in_progress' | 'error';
  message: string;
  steps?: string[];
  timestamp?: string;
  note?: string;
}

interface TrainingHistoryItem {
  timestamp: string;
  backup_path: string;
  files: string[];
}

interface TrainingHistoryResponse {
  total_backups: number;
  backups: TrainingHistoryItem[];
}

const ModelTrainingPanel: React.FC = () => {
  const [status, setStatus] = useState<ModelsStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [training, setTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [history, setHistory] = useState<TrainingHistoryItem[]>([]);
  const [historyExpanded, setHistoryExpanded] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const progressPollRef = useRef<NodeJS.Timeout | null>(null);

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

  // Fetch training progress
  const fetchProgress = useCallback(async () => {
    try {
      const response = await fetch(getApiUrl('/retraining/training-progress'));
      if (response.ok) {
        const data: TrainingProgress = await response.json();
        setTrainingProgress(data);

        // If training just completed, refresh model status
        if (!data.is_training && training) {
          setTraining(false);
          fetchStatus();
        }

        // Update training state based on progress
        if (data.is_training && !training) {
          setTraining(true);
        }

        return data;
      }
    } catch (err) {
      console.error('Failed to fetch training progress:', err);
    }
    return null;
  }, [training, fetchStatus]);

  // Fetch training history
  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch(getApiUrl('/retraining/history'));
      if (response.ok) {
        const data: TrainingHistoryResponse = await response.json();
        setHistory(data.backups || []);
      }
    } catch (err) {
      console.error('Failed to fetch training history:', err);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchProgress(); // Check initial progress on mount

    const interval = setInterval(fetchStatus, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchStatus, fetchProgress]);

  // Fetch history when expanded
  useEffect(() => {
    if (historyExpanded && history.length === 0) {
      fetchHistory();
    }
  }, [historyExpanded, history.length, fetchHistory]);

  // Poll progress while training
  useEffect(() => {
    if (training) {
      // Poll every 2 seconds while training
      progressPollRef.current = setInterval(fetchProgress, 2000);
    } else if (progressPollRef.current) {
      clearInterval(progressPollRef.current);
      progressPollRef.current = null;
    }

    return () => {
      if (progressPollRef.current) {
        clearInterval(progressPollRef.current);
      }
    };
  }, [training, fetchProgress]);

  const triggerTraining = async () => {
    if (!confirm('Start model training? This may take 5-15 minutes and runs in the background.')) {
      return;
    }

    setTraining(true);
    setError(null);

    try {
      const response = await fetch(getApiUrl('/retraining/simple'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const data: TrainingResponse = await response.json();

      if (data.status === 'started' || data.status === 'in_progress') {
        // Start polling for progress
        fetchProgress();
      } else {
        setError(data.message || 'Failed to start training');
        setTraining(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger training');
      setTraining(false);
    }
  };

  const getStepIcon = (stepStatus: string) => {
    switch (stepStatus) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
      case 'running':
        return <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'skipped':
        return <SkipForward className="w-4 h-4 text-gray-400" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStepColor = (stepStatus: string) => {
    switch (stepStatus) {
      case 'completed':
        return 'border-emerald-500/50 bg-emerald-500/10';
      case 'running':
        return 'border-cyan-500/50 bg-cyan-500/10';
      case 'failed':
        return 'border-red-500/50 bg-red-500/10';
      case 'skipped':
        return 'border-gray-500/50 bg-gray-500/10';
      default:
        return 'border-gray-700/50 bg-gray-800/30';
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

  const formatHistoryDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

      if (diffDays === 0) {
        return `Today at ${date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
      } else if (diffDays === 1) {
        return `Yesterday at ${date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
      } else if (diffDays < 7) {
        return `${diffDays} days ago`;
      } else {
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      }
    } catch {
      return isoString;
    }
  };

  const getModelTypesFromFiles = (files: string[]): string[] => {
    const types: string[] = [];
    if (files.some(f => f.includes('model_'))) types.push('Prediction');
    if (files.some(f => f.includes('spike_'))) types.push('Spike');
    if (files.some(f => f.includes('dqn'))) types.push('DQN');
    return types;
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

          {/* Training Progress */}
          {(training || (trainingProgress?.completed_at && !trainingProgress?.error)) && trainingProgress && (
            <div className="mt-4 p-4 bg-gray-800/60 border border-gray-700/50 rounded-xl">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-white">
                  {training ? 'Training in Progress' : 'Training Complete'}
                </span>
                {training && (
                  <span className="text-xs text-cyan-400">
                    Step {trainingProgress.current_step + 1}/{trainingProgress.total_steps}
                  </span>
                )}
              </div>

              {/* Step Progress */}
              <div className="space-y-2">
                {trainingProgress.steps.map((step, index) => (
                  <div
                    key={index}
                    className={`flex items-center gap-3 p-2 rounded-lg border ${getStepColor(step.status)}`}
                  >
                    {getStepIcon(step.status)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <span className={`text-sm ${
                          step.status === 'running' ? 'text-cyan-400' :
                          step.status === 'completed' ? 'text-emerald-400' :
                          step.status === 'failed' ? 'text-red-400' :
                          'text-gray-400'
                        }`}>
                          {step.name}
                        </span>
                        <span className="text-xs text-gray-500 capitalize">
                          {step.status}
                        </span>
                      </div>
                      {step.message && (
                        <p className="text-xs text-gray-400 mt-0.5 truncate">
                          {step.message}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Training Error */}
              {trainingProgress.error && (
                <div className="mt-3 p-2 bg-red-500/10 border border-red-500/30 rounded-lg">
                  <p className="text-xs text-red-400">{trainingProgress.error}</p>
                </div>
              )}

              {/* Completion Time */}
              {trainingProgress.completed_at && !training && (
                <p className="text-xs text-gray-500 mt-3">
                  Completed at {new Date(trainingProgress.completed_at).toLocaleTimeString()}
                </p>
              )}
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {/* Train Button - Always visible */}
          <button
            onClick={triggerTraining}
            disabled={training || !status.data_status.sufficient_for_training}
            className={`mt-4 w-full px-4 py-3 font-medium rounded-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all shadow-lg ${
              status.overall_ready
                ? 'bg-gray-700 hover:bg-gray-600 text-white shadow-gray-900/20'
                : 'bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-400 hover:to-cyan-400 text-white shadow-purple-500/20'
            }`}
          >
            {training ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Starting Training...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                {status.overall_ready ? 'Retrain All Models' : 'Train All Models'}
              </>
            )}
          </button>

          {!status.data_status.sufficient_for_training && (
            <p className="mt-2 text-xs text-center text-amber-400">
              Need at least 50 data records to train models
            </p>
          )}

          {/* Training History Section */}
          <div className="mt-4 border-t border-gray-800/50 pt-4">
            <button
              onClick={() => setHistoryExpanded(!historyExpanded)}
              className="w-full flex items-center justify-between text-sm text-gray-400 hover:text-white transition-colors"
            >
              <div className="flex items-center gap-2">
                <History className="w-4 h-4" />
                <span>Training History</span>
                {history.length > 0 && (
                  <span className="text-xs bg-gray-700 px-1.5 py-0.5 rounded">
                    {history.length}
                  </span>
                )}
              </div>
              {historyExpanded ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {historyExpanded && (
              <div className="mt-3 space-y-2 max-h-48 overflow-y-auto">
                {historyLoading ? (
                  <div className="flex items-center justify-center py-4">
                    <RefreshCw className="w-4 h-4 text-gray-500 animate-spin" />
                  </div>
                ) : history.length === 0 ? (
                  <p className="text-xs text-gray-500 text-center py-4">
                    No training history yet
                  </p>
                ) : (
                  history.slice(0, 10).map((item, index) => {
                    const modelTypes = getModelTypesFromFiles(item.files);
                    return (
                      <div
                        key={index}
                        className="bg-gray-800/40 rounded-lg p-3 border border-gray-700/50"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-gray-300">
                            {formatHistoryDate(item.timestamp)}
                          </span>
                          <span className="text-xs text-gray-500">
                            {item.files.length} files
                          </span>
                        </div>
                        {modelTypes.length > 0 && (
                          <div className="flex gap-1 mt-1">
                            {modelTypes.map(type => (
                              <span
                                key={type}
                                className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400"
                              >
                                {type}
                              </span>
                            ))}
                          </div>
                        )}
                        {item.files.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {item.files.slice(0, 3).map((file, i) => (
                              <span
                                key={i}
                                className="text-[10px] text-gray-500 flex items-center gap-0.5"
                              >
                                <FileText className="w-3 h-3" />
                                {file.replace('.pkl', '')}
                              </span>
                            ))}
                            {item.files.length > 3 && (
                              <span className="text-[10px] text-gray-500">
                                +{item.files.length - 3} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            )}
          </div>
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
