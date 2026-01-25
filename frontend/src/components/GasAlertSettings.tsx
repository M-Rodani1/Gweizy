import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Bell, Plus, Trash2, Power, Check, X, BellOff, BellRing, AlertCircle } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';
import {
  isNotificationSupported,
  getNotificationPermission,
  requestNotificationPermission,
  checkAndTriggerAlerts
} from '../utils/pushNotifications';
import {
  useFormValidation,
  required,
  email,
  positiveNumber,
  discordWebhook,
  numericOnly
} from '../hooks/useFormValidation';

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
  const [notificationPermission, setNotificationPermission] = useState<'granted' | 'denied' | 'default'>('default');
  const [requestingPermission, setRequestingPermission] = useState(false);
  const previousGasRef = useRef<number | null>(null);

  // Form state with separate selects
  const [alertType, setAlertType] = useState<'below' | 'above'>('below');
  const [notificationMethod, setNotificationMethod] = useState('browser');

  // Dynamic validation schema based on notification method
  const validationSchema = useMemo(() => ({
    thresholdGwei: {
      initialValue: '0.001',
      rules: [
        required('Threshold is required'),
        positiveNumber('Threshold must be a positive number')
      ]
    },
    notificationTarget: {
      initialValue: '',
      rules: notificationMethod === 'browser' ? [] : [
        required(`${notificationMethod === 'email' ? 'Email' : notificationMethod === 'discord' ? 'Webhook URL' : 'Chat ID'} is required`),
        ...(notificationMethod === 'email' ? [email('Please enter a valid email address')] : []),
        ...(notificationMethod === 'discord' ? [discordWebhook('Please enter a valid Discord webhook URL')] : []),
        ...(notificationMethod === 'telegram' ? [numericOnly('Chat ID must contain only numbers')] : [])
      ]
    }
  }), [notificationMethod]);

  const {
    values,
    errors,
    touched,
    getFieldProps,
    validateAll,
    reset: resetForm,
    setValue
  } = useFormValidation(validationSchema);

  const userId = walletAddress || 'anonymous';

  // Check notification permission on mount
  useEffect(() => {
    if (isNotificationSupported()) {
      setNotificationPermission(getNotificationPermission());
    }
  }, []);

  useEffect(() => {
    if (userId) {
      fetchAlerts();
    }
  }, [userId]);

  // Check alerts when gas price changes
  useEffect(() => {
    if (alerts.length > 0 && currentGas > 0 && notificationPermission === 'granted') {
      const formattedAlerts = alerts
        .filter(a => a.is_active)
        .map(a => ({
          id: a.id,
          alert_type: a.alert_type as 'below' | 'above',
          threshold_gwei: a.threshold_gwei,
          is_active: a.is_active
        }));

      checkAndTriggerAlerts(currentGas, formattedAlerts, previousGasRef.current);
    }
    previousGasRef.current = currentGas;
  }, [currentGas, alerts, notificationPermission]);

  const fetchAlerts = async () => {
    try {
      const response = await fetch(getApiUrl(`${API_CONFIG.ENDPOINTS.ALERTS}/${userId}`));
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

    // Validate form before submission
    if (!validateAll()) {
      return;
    }

    try {
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.ALERTS), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          alert_type: alertType,
          threshold_gwei: parseFloat(values.thresholdGwei),
          notification_method: notificationMethod,
          notification_target: values.notificationTarget || undefined
        })
      });

      const data = await response.json();

      if (data.success) {
        setAlerts([data.alert, ...alerts]);
        setShowCreateForm(false);
        resetForm();
      }
    } catch (error) {
      console.error('Failed to create alert:', error);
    }
  };

  const toggleAlert = async (alertId: number, currentStatus: boolean) => {
    try {
      const response = await fetch(getApiUrl(`${API_CONFIG.ENDPOINTS.ALERTS}/${alertId}`), {
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
      const response = await fetch(getApiUrl(`${API_CONFIG.ENDPOINTS.ALERTS}/${alertId}`, { user_id: userId }), {
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

  const handleRequestPermission = async () => {
    setRequestingPermission(true);
    const permission = await requestNotificationPermission();
    setNotificationPermission(permission);
    setRequestingPermission(false);
  };

  const sendTestNotification = () => {
    if (notificationPermission !== 'granted') return;

    new Notification('Test Alert', {
      body: `Test notification working! Current gas: ${currentGas.toFixed(4)} gwei`,
      icon: '/logo.svg',
      tag: 'test-notification'
    });
  };

  if (!walletAddress) {
    return (
      <div
        className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl focus-card"
        role="article"
        aria-label="Gas alerts onboarding"
        tabIndex={0}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <Bell className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Gas Price Alerts</h3>
            <p className="text-xs text-gray-400">Connect your wallet to save and sync alerts</p>
          </div>
        </div>
        <p className="text-sm text-gray-300 mb-3">
          We’ll send pings when gas crosses your thresholds. Alerts stay linked to your address.
        </p>
        <div className="flex items-center justify-between text-xs text-gray-400">
          <a href="/docs#alerts" className="text-cyan-300 hover:text-cyan-100 underline decoration-dotted">
            Learn how alerts work
          </a>
          <span>Use the header wallet button to connect</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl focus-card"
      role="article"
      aria-label="Gas price alerts"
      tabIndex={0}
    >
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
        <a
          href="/docs#alerts"
          className="text-xs text-cyan-300 hover:text-cyan-100 underline decoration-dotted mr-3"
        >
          Learn more
        </a>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="px-4 py-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
        >
          {showCreateForm ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
          {showCreateForm ? 'Cancel' : 'New Alert'}
        </button>
      </div>

      {/* Notification Permission Banner */}
      {isNotificationSupported() && notificationPermission !== 'granted' && (
        <div className={`mb-6 p-4 rounded-lg border flex items-center justify-between ${
          notificationPermission === 'denied'
            ? 'bg-red-500/10 border-red-500/30'
            : 'bg-yellow-500/10 border-yellow-500/30'
        }`}>
          <div className="flex items-center gap-3">
            {notificationPermission === 'denied' ? (
              <BellOff className="w-5 h-5 text-red-400" />
            ) : (
              <BellRing className="w-5 h-5 text-yellow-400" />
            )}
            <div>
              <p className={`text-sm font-medium ${
                notificationPermission === 'denied' ? 'text-red-400' : 'text-yellow-400'
              }`}>
                {notificationPermission === 'denied'
                  ? 'Notifications Blocked'
                  : 'Enable Push Notifications'}
              </p>
              <p className="text-xs text-gray-400">
                {notificationPermission === 'denied'
                  ? 'Enable in browser settings to receive alerts'
                  : 'Get notified even when the browser is in the background'}
              </p>
            </div>
          </div>
          {notificationPermission === 'default' && (
            <button
              onClick={handleRequestPermission}
              disabled={requestingPermission}
              className="px-4 py-2 bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-400 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              {requestingPermission ? 'Requesting...' : 'Enable'}
            </button>
          )}
        </div>
      )}

      {/* Notification Enabled Banner */}
      {notificationPermission === 'granted' && (
        <div className="mb-6 p-3 bg-green-500/10 border border-green-500/30 rounded-lg flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BellRing className="w-4 h-4 text-green-400" />
            <span className="text-sm text-green-400">Push notifications enabled</span>
          </div>
          <button
            onClick={sendTestNotification}
            className="text-xs text-green-400 hover:text-green-300 underline"
          >
            Send test
          </button>
        </div>
      )}

      {/* Create Form */}
      {showCreateForm && (
        <form onSubmit={createAlert} className="mb-6 p-4 bg-slate-700/30 rounded-lg border border-slate-600" aria-label="Create new gas price alert">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label htmlFor="alert-type" className="block text-sm font-medium text-gray-300 mb-2">Alert Type</label>
              <select
                id="alert-type"
                value={alertType}
                onChange={(e) => setAlertType(e.target.value as 'below' | 'above')}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value="below">Notify when below</option>
                <option value="above">Notify when above</option>
              </select>
            </div>
            <div>
              <label htmlFor="threshold-gwei" className="block text-sm font-medium text-gray-300 mb-2">
                Threshold (gwei)
                {currentGas > 0 && (
                  <button
                    type="button"
                    onClick={() => setValue('thresholdGwei', getSuggestion())}
                    className="ml-2 text-xs text-blue-400 hover:text-blue-300 focus:outline-none focus:underline"
                    aria-label={`Use suggested threshold of ${getSuggestion()} gwei`}
                  >
                    Suggest: {getSuggestion()}
                  </button>
                )}
              </label>
              <input
                id="threshold-gwei"
                type="number"
                step="0.0001"
                {...getFieldProps('thresholdGwei')}
                className={`w-full px-3 py-2 bg-slate-800 border rounded-lg text-white focus:outline-none focus:ring-2 transition-colors ${
                  touched.thresholdGwei && errors.thresholdGwei
                    ? 'border-red-500 focus:border-red-500 focus:ring-red-500/50'
                    : 'border-slate-600 focus:border-blue-500 focus:ring-blue-500/50'
                }`}
                placeholder="0.001"
                aria-required="true"
              />
              {touched.thresholdGwei && errors.thresholdGwei && (
                <p id="thresholdGwei-error" className="mt-1 text-xs text-red-400 flex items-center gap-1" role="alert">
                  <AlertCircle className="w-3 h-3" aria-hidden="true" />
                  {errors.thresholdGwei}
                </p>
              )}
            </div>
          </div>

          {/* Notification Method Selection */}
          <div className="mb-4">
            <label htmlFor="notification-method" className="block text-sm font-medium text-gray-300 mb-2">Notification Method</label>
            <select
              id="notification-method"
              value={notificationMethod}
              onChange={(e) => {
                setNotificationMethod(e.target.value);
                setValue('notificationTarget', '');  // Clear target when method changes
              }}
              className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            >
              <option value="browser">Browser Notification</option>
              <option value="email">Email</option>
              <option value="discord">Discord Webhook</option>
              <option value="telegram">Telegram</option>
            </select>
          </div>

          {/* Notification Target Input (for email/discord/telegram) */}
          {notificationMethod !== 'browser' && (
            <div className="mb-4">
              <label htmlFor="notification-target" className="block text-sm font-medium text-gray-300 mb-2">
                {notificationMethod === 'email' && 'Email Address'}
                {notificationMethod === 'discord' && 'Discord Webhook URL'}
                {notificationMethod === 'telegram' && 'Telegram Chat ID'}
              </label>
              <input
                id="notification-target"
                type={notificationMethod === 'email' ? 'email' : 'text'}
                {...getFieldProps('notificationTarget')}
                className={`w-full px-3 py-2 bg-slate-800 border rounded-lg text-white focus:outline-none focus:ring-2 transition-colors ${
                  touched.notificationTarget && errors.notificationTarget
                    ? 'border-red-500 focus:border-red-500 focus:ring-red-500/50'
                    : 'border-slate-600 focus:border-blue-500 focus:ring-blue-500/50'
                }`}
                placeholder={
                  notificationMethod === 'email' ? 'your@email.com' :
                  notificationMethod === 'discord' ? 'https://discord.com/api/webhooks/...' :
                  '123456789'  // Telegram chat ID
                }
                aria-required="true"
                aria-describedby={
                  touched.notificationTarget && errors.notificationTarget
                    ? 'notificationTarget-error'
                    : notificationMethod === 'telegram' ? 'telegram-help' : undefined
                }
              />
              {touched.notificationTarget && errors.notificationTarget ? (
                <p id="notificationTarget-error" className="mt-1 text-xs text-red-400 flex items-center gap-1" role="alert">
                  <AlertCircle className="w-3 h-3" aria-hidden="true" />
                  {errors.notificationTarget}
                </p>
              ) : notificationMethod === 'telegram' ? (
                <p id="telegram-help" className="text-xs text-gray-400 mt-1">
                  Get your chat ID by messaging @userinfobot on Telegram
                </p>
              ) : null}
            </div>
          )}

          <button
            type="submit"
            className="w-full px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-800"
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
                <div className="flex items-center gap-2" role="group" aria-label="Alert actions">
                  <button
                    onClick={() => toggleAlert(alert.id, alert.is_active)}
                    className={`p-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 ${
                      alert.is_active
                        ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30 focus:ring-green-500'
                        : 'bg-gray-500/20 text-gray-400 hover:bg-gray-500/30 focus:ring-gray-500'
                    }`}
                    aria-label={`${alert.is_active ? 'Disable' : 'Enable'} alert for ${alert.alert_type} ${alert.threshold_gwei.toFixed(4)} gwei`}
                    aria-pressed={alert.is_active}
                  >
                    <Power className="w-4 h-4" aria-hidden="true" />
                  </button>
                  <button
                    onClick={() => deleteAlert(alert.id)}
                    className="p-2 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-slate-800"
                    aria-label={`Delete alert for ${alert.alert_type} ${alert.threshold_gwei.toFixed(4)} gwei`}
                  >
                    <Trash2 className="w-4 h-4" aria-hidden="true" />
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

export default React.memo(GasAlertSettings);
