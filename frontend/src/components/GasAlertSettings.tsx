import React, { useState, useEffect } from 'react';
import { Bell, Plus, Trash2, Power, Check, X } from 'lucide-react';

interface Alert {
  id: number;
  alert_type: string;
  threshold_gwei: number;
  notification_method: string;
  is_active: boolean;
  last_triggered: string | null;
  created_at: string;
}

interface GasAlertSettingsProps {
  currentGas: number;
  walletAddress?: string | null;
}

const GasAlertSettings: React.FC<GasAlertSettingsProps> = ({ currentGas, walletAddress }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateForm, setShowCreateForm] = useState(false);

  // Form state
  const [alertType, setAlertType] = useState<'below' | 'above'>('below');
  const [thresholdGwei, setThresholdGwei] = useState('0.001');
  const [notificationMethod, setNotificationMethod] = useState('browser');

  const API_BASE = import.meta.env.VITE_API_URL || 'https://basegasfeesml-production.up.railway.app/api';
  const userId = walletAddress || 'anonymous';

  useEffect(() => {
    if (userId) {
      fetchAlerts();
    }
  }, [userId]);

  const fetchAlerts = async () => {
    try {
      const response = await fetch(`${API_BASE}/alerts/${userId}`);
      const data = await response.json();

      if (data.success) {
        setAlerts(data.alerts);
      }
      setLoading(false);
    } catch (error) {
      setLoading(false);
    }
  };

  const createAlert = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const response = await fetch(`${API_BASE}/alerts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          alert_type: alertType,
          threshold_gwei: parseFloat(thresholdGwei),
          notification_method: notificationMethod
        })
      });

      const data = await response.json();

      if (data.success) {
        setAlerts([data.alert, ...alerts]);
        setShowCreateForm(false);
        setThresholdGwei('0.001');
      }
    } catch (error) {
      console.error('Failed to create alert:', error);
    }
  };

  const toggleAlert = async (alertId: number, currentStatus: boolean) => {
    try {
      const response = await fetch(`${API_BASE}/alerts/${alertId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_active: !currentStatus })
      });

      const data = await response.json();

      if (data.success) {
        setAlerts(alerts.map(a =>
          a.id === alertId ? { ...a, is_active: !currentStatus } : a
        ));
      }
    } catch (error) {
      console.error('Failed to toggle alert:', error);
    }
  };

  const deleteAlert = async (alertId: number) => {
    if (!confirm('Are you sure you want to delete this alert?')) return;

    try {
      const response = await fetch(`${API_BASE}/alerts/${alertId}?user_id=${userId}`, {
        method: 'DELETE'
      });

      const data = await response.json();

      if (data.success) {
        setAlerts(alerts.filter(a => a.id !== alertId));
      }
    } catch (error) {
      console.error('Failed to delete alert:', error);
    }
  };

  const getSuggestion = () => {
    if (currentGas === 0) return '0.001';
    return (currentGas * 0.8).toFixed(4); // Suggest 20% below current
  };

  return (
    <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <Bell className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Gas Price Alerts</h3>
            <p className="text-xs text-gray-400">Get notified when gas reaches your target</p>
          </div>
        </div>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
        >
          {showCreateForm ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
          {showCreateForm ? 'Cancel' : 'New Alert'}
        </button>
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <form onSubmit={createAlert} className="mb-6 p-4 bg-slate-700/30 rounded-lg border border-slate-600">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Alert Type</label>
              <select
                value={alertType}
                onChange={(e) => setAlertType(e.target.value as 'below' | 'above')}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="below">Notify when below</option>
                <option value="above">Notify when above</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Threshold (gwei)
                {currentGas > 0 && (
                  <button
                    type="button"
                    onClick={() => setThresholdGwei(getSuggestion())}
                    className="ml-2 text-xs text-blue-400 hover:text-blue-300"
                  >
                    Suggest: {getSuggestion()}
                  </button>
                )}
              </label>
              <input
                type="number"
                step="0.0001"
                value={thresholdGwei}
                onChange={(e) => setThresholdGwei(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                placeholder="0.001"
                required
              />
            </div>
          </div>
          <button
            type="submit"
            className="w-full px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
          >
            Create Alert
          </button>
        </form>
      )}

      {/* Alerts List */}
      {loading ? (
        <div className="text-center py-8 text-gray-400">Loading alerts...</div>
      ) : alerts.length === 0 ? (
        <div className="text-center py-8 text-gray-400">
          <Bell className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No alerts yet. Create one to get started!</p>
        </div>
      ) : (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-4 rounded-lg border transition-all ${
                alert.is_active
                  ? 'bg-slate-700/30 border-slate-600'
                  : 'bg-slate-800/30 border-slate-700 opacity-60'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`font-semibold ${alert.is_active ? 'text-white' : 'text-gray-400'}`}>
                      {alert.alert_type === 'below' ? '↓ Below' : '↑ Above'} {alert.threshold_gwei.toFixed(4)} gwei
                    </span>
                    {alert.is_active && (
                      <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full border border-green-500/30">
                        Active
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-400">
                    {alert.last_triggered ? (
                      `Last triggered: ${new Date(alert.last_triggered).toLocaleDateString()}`
                    ) : (
                      'Never triggered'
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => toggleAlert(alert.id, alert.is_active)}
                    className={`p-2 rounded-lg transition-colors ${
                      alert.is_active
                        ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                        : 'bg-gray-500/20 text-gray-400 hover:bg-gray-500/30'
                    }`}
                    title={alert.is_active ? 'Disable' : 'Enable'}
                  >
                    <Power className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => deleteAlert(alert.id)}
                    className="p-2 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-lg transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Current Gas Info */}
      {currentGas > 0 && (
        <div className="mt-6 p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Current Gas Price:</span>
            <span className="font-bold text-white">{currentGas.toFixed(4)} gwei</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default GasAlertSettings;
