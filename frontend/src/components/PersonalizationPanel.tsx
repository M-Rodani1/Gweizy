import React from 'react';
import { Sliders, Sparkles, Target, Zap } from 'lucide-react';
import { TX_TYPE_META } from '../config/transactions';
import { TransactionType } from '../config/chains';
import { usePreferences } from '../contexts/PreferencesContext';

const STRATEGY_OPTIONS = [
  {
    id: 'saver',
    label: 'Saver',
    description: 'Lower urgency, wait for bigger drops'
  },
  {
    id: 'balanced',
    label: 'Balanced',
    description: 'Best of cost + speed'
  },
  {
    id: 'fast',
    label: 'Fast',
    description: 'Prioritize confirmation speed'
  }
] as const;

const EXPIRY_OPTIONS = [6, 12, 24, 48, 72];

const PersonalizationPanel: React.FC = () => {
  const { preferences, setStrategy, updatePreferences } = usePreferences();

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
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-5 shadow-xl h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Sliders className="w-4 h-4 text-cyan-400" />
        <h3 className="font-semibold text-white">Profile & Defaults</h3>
      </div>

      <div className="mb-5">
        <div className="text-xs text-gray-400 mb-2">Strategy preset</div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {STRATEGY_OPTIONS.map((option) => (
            <button
              key={option.id}
              onClick={() => setStrategy(option.id)}
              className={`rounded-xl border px-3 py-3 text-left transition ${
                preferences.strategy === option.id
                  ? 'border-cyan-500/60 bg-cyan-500/10 text-cyan-200'
                  : 'border-gray-700 bg-gray-900/40 text-gray-300 hover:border-gray-600'
              }`}
            >
              <div className="flex items-center gap-2 font-medium">
                {option.id === 'saver' && <Target className="w-4 h-4 text-emerald-400" />}
                {option.id === 'balanced' && <Sparkles className="w-4 h-4 text-cyan-400" />}
                {option.id === 'fast' && <Zap className="w-4 h-4 text-amber-400" />}
                {option.label}
              </div>
              <div className="text-xs text-gray-400 mt-1">{option.description}</div>
            </button>
          ))}
        </div>
        {preferences.strategy === 'custom' && (
          <div className="text-xs text-amber-300 mt-2">
            Custom profile active
          </div>
        )}
      </div>

      <div className="mb-5">
        <div className="text-xs text-gray-400 mb-2">Default transaction type</div>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(TX_TYPE_META).slice(0, 6).map(([type, meta]) => {
            const Icon = meta.icon;
            return (
              <button
                key={type}
                onClick={() => handleTxTypeChange(type as TransactionType)}
                className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm transition ${
                  preferences.defaultTxType === type
                    ? 'border-cyan-500/60 bg-cyan-500/10 text-cyan-200'
                    : 'border-gray-700 bg-gray-900/40 text-gray-300 hover:border-gray-600'
                }`}
              >
                <Icon className="w-4 h-4" />
                {meta.shortLabel}
              </button>
            );
          })}
        </div>
      </div>

      <div className="mb-5">
        <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
          <span>Default urgency</span>
          <span className="text-gray-300">{Math.round(preferences.urgency * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={preferences.urgency}
          onChange={(e) => handleUrgencyChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
        />
      </div>

      <div className="border-t border-gray-800 pt-4">
        <div className="text-xs text-gray-400 mb-2">Scheduling defaults</div>
        <div className="grid grid-cols-3 gap-2 text-xs text-gray-400 mb-3">
          <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-2">
            Target discount
            <div className="text-sm text-white mt-1">
              {Math.round((1 - preferences.schedule.targetMultiplier) * 100)}%
            </div>
          </div>
          <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-2">
            Max buffer
            <div className="text-sm text-white mt-1">
              {Math.round((preferences.schedule.maxMultiplier - 1) * 100)}%
            </div>
          </div>
          <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-2">
            Expiry
            <div className="text-sm text-white mt-1">{preferences.schedule.expiryHours}h</div>
          </div>
        </div>

        <div className="space-y-3">
          <div>
            <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
              <span>Target multiplier</span>
              <span className="text-gray-300">
                {preferences.schedule.targetMultiplier.toFixed(2)}x
              </span>
            </div>
            <input
              type="range"
              min="0.75"
              max="0.98"
              step="0.01"
              value={preferences.schedule.targetMultiplier}
              onChange={(e) => handleScheduleChange({ targetMultiplier: parseFloat(e.target.value) })}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-emerald-400"
            />
          </div>
          <div>
            <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
              <span>Max gas multiplier</span>
              <span className="text-gray-300">
                {preferences.schedule.maxMultiplier.toFixed(2)}x
              </span>
            </div>
            <input
              type="range"
              min="1.02"
              max="1.3"
              step="0.02"
              value={preferences.schedule.maxMultiplier}
              onChange={(e) => handleScheduleChange({ maxMultiplier: parseFloat(e.target.value) })}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-400"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-2 block">Default expiry</label>
            <select
              value={preferences.schedule.expiryHours}
              onChange={(e) => handleScheduleChange({ expiryHours: parseInt(e.target.value, 10) })}
              className="w-full rounded-lg bg-gray-900/60 border border-gray-800 text-gray-200 px-3 py-2 text-sm"
            >
              {EXPIRY_OPTIONS.map((hours) => (
                <option key={hours} value={hours}>
                  {hours} hours
                </option>
              ))}
            </select>
          </div>
        </div>

        <label className="mt-4 flex items-center gap-2 text-xs text-gray-400">
          <input
            type="checkbox"
            checked={preferences.showAdvancedFields}
            onChange={(e) => updatePreferences({ showAdvancedFields: e.target.checked })}
            className="h-4 w-4 rounded border-gray-600 bg-gray-800 text-cyan-400"
          />
          Show advanced transaction fields
        </label>
      </div>
    </div>
  );
};

export default PersonalizationPanel;
