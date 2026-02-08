import React, { useEffect, useState } from 'react';
import { Activity, CloudOff, RefreshCw, ShieldCheck, Brain, AlertTriangle } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';
import type { DriftInfo } from '../types/modelMetrics';

type StatusState = 'checking' | 'online' | 'offline' | 'degraded';

interface StatusItem {
  key: string;
  label: string;
  status: StatusState;
  detail: string;
}

const ApiStatusPanel: React.FC = () => {
  const [items, setItems] = useState<StatusItem[]>([]);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  const fetchStatus = async () => {
    const now = new Date();
    setLastUpdated(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));

    const baseItems: StatusItem[] = [
      { key: 'health', label: 'Core API', status: 'checking', detail: 'Checking health...' },
      { key: 'agent', label: 'AI Agent', status: 'checking', detail: 'Checking agent...' },
      { key: 'model', label: 'Model Health', status: 'checking', detail: 'Checking accuracy...' },
      { key: 'patterns', label: 'Patterns', status: 'checking', detail: 'Checking patterns...' }
    ];
    setItems(baseItems);

    try {
      const [healthRes, agentRes, driftRes, patternsRes] = await Promise.all([
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.HEALTH)),
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.AGENT_STATUS)),
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_DRIFT)),
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.GAS_PATTERNS))
      ]);

      const nextItems = [...baseItems];

      if (healthRes.ok) {
        nextItems[0] = { key: 'health', label: 'Core API', status: 'online', detail: 'Streaming live data' };
      } else {
        nextItems[0] = { key: 'health', label: 'Core API', status: 'offline', detail: 'Health check failed' };
      }

      if (agentRes.ok) {
        const agentData = await agentRes.json();
        // Check if agent is loaded - dqn_loaded is the actual response field
        const agentLoaded = agentData?.dqn_loaded ?? true;
        const modelType = agentData?.model_type ?? 'Unknown';
        nextItems[1] = {
          key: 'agent',
          label: 'AI Agent',
          status: agentLoaded ? 'online' : 'degraded',
          detail: agentLoaded ? `${modelType} agent online` : 'Fallback to heuristic'
        };
      } else {
        nextItems[1] = { key: 'agent', label: 'AI Agent', status: 'offline', detail: 'Agent unreachable' };
      }

      // Model health / drift check
      if (driftRes.ok) {
        const driftData = await driftRes.json();
        const shouldRetrain = driftData?.should_retrain ?? false;
        const isDrifting = Object.values(driftData?.drift || {}).some((d) => (d as DriftInfo)?.is_drifting);

        if (shouldRetrain) {
          nextItems[2] = { key: 'model', label: 'Model Health', status: 'degraded', detail: 'Retrain recommended' };
        } else if (isDrifting) {
          nextItems[2] = { key: 'model', label: 'Model Health', status: 'degraded', detail: 'Drift detected' };
        } else {
          nextItems[2] = { key: 'model', label: 'Model Health', status: 'online', detail: 'Accuracy stable' };
        }
      } else {
        nextItems[2] = { key: 'model', label: 'Model Health', status: 'degraded', detail: 'No tracking data' };
      }

      if (patternsRes.ok) {
        nextItems[3] = { key: 'patterns', label: 'Patterns', status: 'online', detail: 'Hourly & daily patterns ready' };
      } else if (patternsRes.status === 404 || patternsRes.status === 503) {
        nextItems[3] = { key: 'patterns', label: 'Patterns', status: 'degraded', detail: 'Using cached patterns' };
      } else {
        nextItems[3] = { key: 'patterns', label: 'Patterns', status: 'offline', detail: 'Patterns unavailable' };
      }

      setItems(nextItems);
    } catch (err) {
      setItems([
        { key: 'health', label: 'Core API', status: 'offline', detail: 'Unable to reach API' },
        { key: 'agent', label: 'AI Agent', status: 'offline', detail: 'Agent unavailable' },
        { key: 'model', label: 'Model Health', status: 'offline', detail: 'Tracking unavailable' },
        { key: 'patterns', label: 'Patterns', status: 'offline', detail: 'Patterns unavailable' }
      ]);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  const getStatusStyle = (status: StatusState) => {
    switch (status) {
      case 'online':
        return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30';
      case 'degraded':
        return 'text-amber-400 bg-amber-500/10 border-amber-500/30';
      case 'offline':
        return 'text-red-400 bg-red-500/10 border-red-500/30';
      default:
        return 'text-gray-400 bg-gray-500/10 border-gray-500/30';
    }
  };

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl widget-glow h-full flex flex-col bg-pattern-grid">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-cyan-400" />
          <h3 className="font-semibold text-white">System Status</h3>
        </div>
        <button
          onClick={fetchStatus}
          className="btn-gradient-secondary text-xs text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-medium"
        >
          <RefreshCw className="w-3 h-3" />
          Refresh
        </button>
      </div>

      <div className="space-y-3">
        {items.map((item) => (
          <div
            key={item.key}
            className={`flex items-center justify-between rounded-xl border px-3 py-2 ${getStatusStyle(item.status)}`}
          >
            <div className="flex items-center gap-2">
              {item.status === 'offline' ? (
                <CloudOff className="w-4 h-4" />
              ) : item.key === 'model' ? (
                item.status === 'degraded' ? (
                  <AlertTriangle className="w-4 h-4" />
                ) : (
                  <Brain className="w-4 h-4" />
                )
              ) : (
                <Activity className="w-4 h-4" />
              )}
              <span className="text-sm font-medium text-white">{item.label}</span>
            </div>
            <span className="text-xs text-gray-200">{item.detail}</span>
          </div>
        ))}
      </div>

      <div className="mt-auto pt-4 border-t border-gray-800/50">
        <div className="last-updated text-xs">
          Last check: {lastUpdated || '...'}
        </div>
      </div>
    </div>
  );
};

export default ApiStatusPanel;
