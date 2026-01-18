import React, { useState } from 'react';
import { Sliders, Sparkles, Target, Zap, ChevronDown, ChevronUp } from 'lucide-react';
import { TX_TYPE_META } from '../config/transactions';
import { TransactionType } from '../config/chains';
import { usePreferences } from '../contexts/PreferencesContext';

const STRATEGY_OPTIONS = [
  {
    id: 'saver',
    label: 'Saver',
    description: 'Wait for drops'
  },
  {
    id: 'balanced',
    label: 'Balanced',
    description: 'Cost + speed'
  },
  {
    id: 'fast',
    label: 'Fast',
    description: 'Speed priority'
  }
] as const;

const EXPIRY_OPTIONS = [6, 12, 24, 48, 72];

const PersonalizationPanel: React.FC = () => {
  const { preferences, setStrategy, updatePreferences } = usePreferences();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleTxTypeChange = (type: TransactionType) => {
    updatePreferences({ defaultTxType: type, strategy: 'custom' });
  };

  const handleUrgencyChange = (value: number) => {
    updatePreferences({ urgency: value, strategy: 'custom' });
  };

  const handleScheduleChange = (updates: Partial<typeof preferences.schedule>) => {
    updatePreferences({
      schedule: { ...preferences.schedule, ...updates },
      strategy: 'custom'
    });
  };

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-5 shadow-xl widget-glow h-full flex flex-col w-full max-w-full overflow-hidden">
      <div className="flex items-center gap-2 mb-4">
        <Sliders className="w-4 h-4 text-cyan-400" />
        <h3 className="font-semibold text-white text-sm">Profile & Defaults</h3>
      </div>

      {/* Strategy Presets - Compact */}
      <div className="mb-4">
        <div className="text-xs text-gray-500 mb-2">Strategy</div>
        <div className="flex gap-2">
          {STRATEGY_OPTIONS.map((option) => (
            <button
              key={option.id}
              onClick={() => setStrategy(option.id)}
              className={`flex-1 rounded-lg border px-2 py-2 text-center transition ${
                preferences.strategy === option.id
                  ? 'border-cyan-500/60 bg-cyan-500/10 text-cyan-200'
                  : 'border-gray-700 bg-gray-900/40 text-gray-400 hover:border-gray-600'
              }`}
            >
              <div className="flex items-center justify-center gap-1.5">
                {option.id === 'saver' && <Target className="w-3.5 h-3.5 text-emerald-400" />}
                {option.id === 'balanced' && <Sparkles className="w-3.5 h-3.5 text-cyan-400" />}
                {option.id === 'fast' && <Zap className="w-3.5 h-3.5 text-amber-400" />}
                <span className="text-xs font-medium">{option.label}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Transaction Type - Compact */}
      <div className="mb-4">
        <div className="text-xs text-gray-500 mb-2">Default Transaction</div>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(TX_TYPE_META).slice(0, 5).map(([type, meta]) => {
            const Icon = meta.icon;
            return (
              <button
                key={type}
                onClick={() => handleTxTypeChange(type as TransactionType)}
                className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs transition ${
                  preferences.defaultTxType === type
                    ? 'border-cyan-500/60 bg-cyan-500/10 text-cyan-200'
                    : 'border-gray-700 bg-gray-900/40 text-gray-400 hover:border-gray-600'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {meta.shortLabel}
              </button>
            );
          })}
        </div>
      </div>

      {/* Urgency Slider - Compact */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="text-gray-500">Urgency</span>
          <span className="text-cyan-400 font-mono">{Math.round(preferences.urgency * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={preferences.urgency}
          onChange={(e) => handleUrgencyChange(parseFloat(e.target.value))}
          className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
        />
      </div>

      {/* Advanced Settings Toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center justify-between w-full text-xs text-gray-500 hover:text-gray-300 transition-colors py-2 border-t border-gray-800"
      >
        <span>Advanced settings</span>
        {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>

      {/* Advanced Settings - Collapsible */}
      {showAdvanced && (
        <div className="space-y-3 pt-2 animate-fadeIn">
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-2 text-center">
              <div className="text-gray-500">Discount</div>
              <div className="text-white font-medium">
                {Math.round((1 - preferences.schedule.targetMultiplier) * 100)}%
              </div>
            </div>
            <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-2 text-center">
              <div className="text-gray-500">Buffer</div>
              <div className="text-white font-medium">
                {Math.round((preferences.schedule.maxMultiplier - 1) * 100)}%
              </div>
            </div>
            <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-2 text-center">
              <div className="text-gray-500">Expiry</div>
              <div className="text-white font-medium">{preferences.schedule.expiryHours}h</div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
              <span>Target multiplier</span>
              <span className="text-gray-300">{preferences.schedule.targetMultiplier.toFixed(2)}x</span>
            </div>
            <input
              type="range"
              min="0.75"
              max="0.98"
              step="0.01"
              value={preferences.schedule.targetMultiplier}
              onChange={(e) => handleScheduleChange({ targetMultiplier: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-emerald-400"
            />
          </div>

          <div>
            <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
              <span>Max gas multiplier</span>
              <span className="text-gray-300">{preferences.schedule.maxMultiplier.toFixed(2)}x</span>
            </div>
            <input
              type="range"
              min="1.02"
              max="1.3"
              step="0.02"
              value={preferences.schedule.maxMultiplier}
              onChange={(e) => handleScheduleChange({ maxMultiplier: parseFloat(e.target.value) })}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-400"
            />
          </div>

          <select
            value={preferences.schedule.expiryHours}
            onChange={(e) => handleScheduleChange({ expiryHours: parseInt(e.target.value, 10) })}
            className="w-full rounded-lg bg-gray-900/60 border border-gray-800 text-gray-200 px-3 py-1.5 text-xs"
          >
            {EXPIRY_OPTIONS.map((hours) => (
              <option key={hours} value={hours}>
                Expire after {hours} hours
              </option>
            ))}
          </select>

          <label className="flex items-center gap-2 text-xs text-gray-500">
            <input
              type="checkbox"
              checked={preferences.showAdvancedFields}
              onChange={(e) => updatePreferences({ showAdvancedFields: e.target.checked })}
              className="h-3.5 w-3.5 rounded border-gray-600 bg-gray-800 text-cyan-400"
            />
            Show advanced transaction fields
          </label>
        </div>
      )}
    </div>
  );
};

export default PersonalizationPanel;
