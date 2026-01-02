import React, { useEffect, useState } from 'react';
import { AlertTriangle, X, RefreshCw, TrendingDown } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface DriftInfo {
  horizon: string;
  driftRatio: number;
  currentMae: number;
  baselineMae: number;
}

interface DriftAlertBannerProps {
  onDismiss?: () => void;
  checkInterval?: number; // ms between checks
}

const DriftAlertBanner: React.FC<DriftAlertBannerProps> = ({
  onDismiss,
  checkInterval = 300000 // 5 minutes
}) => {
  const [driftData, setDriftData] = useState<DriftInfo[]>([]);
  const [shouldRetrain, setShouldRetrain] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [retrainStatus, setRetrainStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const checkDrift = async () => {
    try {
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_DRIFT));

      if (!response.ok) {
        setLoading(false);
        return;
      }

      const data = await response.json();

      if (data.success) {
        const driftingHorizons: DriftInfo[] = [];

        Object.entries(data.drift || {}).forEach(([horizon, info]: [string, any]) => {
          if (info?.is_drifting) {
            driftingHorizons.push({
              horizon,
              driftRatio: info.drift_ratio || 0,
              currentMae: info.mae_current || 0,
              baselineMae: info.mae_baseline || 0
            });
          }
        });

        setDriftData(driftingHorizons);
        setShouldRetrain(data.should_retrain || false);
      }
    } catch (err) {
      // Silently fail - don't show alert on error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkDrift();
    const interval = setInterval(checkDrift, checkInterval);
    return () => clearInterval(interval);
  }, [checkInterval]);

  const handleDismiss = () => {
    setDismissed(true);
    onDismiss?.();
  };

  const handleTriggerRetrain = async () => {
    if (retraining) return;

    setRetraining(true);
    setRetrainStatus('idle');

    try {
      const response = await fetch(getApiUrl('/retraining/simple'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        setRetrainStatus('success');
        // Auto-dismiss after successful retrain
        setTimeout(() => {
          setDismissed(true);
          onDismiss?.();
        }, 3000);
      } else {
        setRetrainStatus('error');
      }
    } catch (err) {
      setRetrainStatus('error');
    } finally {
      setRetraining(false);
    }
  };

  // Don't show if dismissed, loading, or no drift detected
  if (dismissed || loading || (driftData.length === 0 && !shouldRetrain)) {
    return null;
  }

  const severity = shouldRetrain ? 'high' : 'medium';
  const bgColor = severity === 'high'
    ? 'bg-gradient-to-r from-red-900/90 to-red-800/90'
    : 'bg-gradient-to-r from-amber-900/90 to-amber-800/90';
  const borderColor = severity === 'high'
    ? 'border-red-500/50'
    : 'border-amber-500/50';
  const iconColor = severity === 'high'
    ? 'text-red-400'
    : 'text-amber-400';

  return (
    <div
      className={`${bgColor} ${borderColor} border rounded-xl p-4 mb-4 shadow-lg backdrop-blur-sm animate-fade-in`}
    >
      <div className="flex items-start justify-between gap-4">
        {/* Icon and Content */}
        <div className="flex items-start gap-3">
          <div className={`p-2 rounded-lg ${severity === 'high' ? 'bg-red-500/20' : 'bg-amber-500/20'}`}>
            <AlertTriangle className={`w-5 h-5 ${iconColor}`} />
          </div>

          <div className="flex-1">
            <h4 className="font-semibold text-white flex items-center gap-2">
              {shouldRetrain ? 'Model Retrain Recommended' : 'Model Drift Detected'}
              {severity === 'high' && (
                <span className="px-1.5 py-0.5 text-xs bg-red-500/30 text-red-300 rounded">
                  Action Needed
                </span>
              )}
            </h4>

            <p className="text-sm text-gray-300 mt-1">
              {shouldRetrain
                ? 'Prediction accuracy has degraded significantly. Retraining is recommended for optimal performance.'
                : `Performance drift detected in ${driftData.map(d => d.horizon).join(', ')} predictions.`}
            </p>

            {/* Drift Details */}
            {driftData.length > 0 && (
              <div className="flex flex-wrap gap-3 mt-3">
                {driftData.map(drift => (
                  <div
                    key={drift.horizon}
                    className="flex items-center gap-2 px-2 py-1 bg-black/20 rounded-lg text-xs"
                  >
                    <TrendingDown className="w-3 h-3 text-red-400" />
                    <span className="text-gray-300">{drift.horizon}:</span>
                    <span className="text-red-400 font-mono">
                      +{(drift.driftRatio * 100).toFixed(0)}% error
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Actions */}
            <div className="flex flex-wrap items-center gap-2 sm:gap-3 mt-3">
              <button
                onClick={() => window.location.href = '/analytics'}
                className="text-xs px-3 py-2 min-h-[36px] bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors flex items-center gap-1"
              >
                View Details
              </button>
              {shouldRetrain && (
                <button
                  onClick={handleTriggerRetrain}
                  disabled={retraining}
                  className={`text-xs px-3 py-2 min-h-[36px] text-white rounded-lg transition-colors flex items-center gap-1 ${
                    retraining
                      ? 'bg-purple-500/50 cursor-wait'
                      : retrainStatus === 'success'
                        ? 'bg-emerald-500'
                        : retrainStatus === 'error'
                          ? 'bg-red-500'
                          : 'bg-purple-500/80 hover:bg-purple-500'
                  }`}
                >
                  <RefreshCw className={`w-3 h-3 ${retraining ? 'animate-spin' : ''}`} />
                  <span className="hidden xs:inline">
                    {retraining
                      ? 'Retraining...'
                      : retrainStatus === 'success'
                        ? 'Retrained!'
                        : retrainStatus === 'error'
                          ? 'Failed - Retry'
                          : 'Trigger Retrain'
                    }
                  </span>
                  <span className="xs:hidden">
                    {retraining ? '...' : retrainStatus === 'success' ? '✓' : retrainStatus === 'error' ? '!' : 'Retrain'}
                  </span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Dismiss Button */}
        <button
          onClick={handleDismiss}
          className="p-1 text-gray-400 hover:text-white transition-colors rounded-lg hover:bg-white/10"
          aria-label="Dismiss alert"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default DriftAlertBanner;
