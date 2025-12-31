import React, { createContext, useContext, useEffect, useMemo, useState, ReactNode } from 'react';
import { TransactionType } from '../config/chains';

export type StrategyProfile = 'saver' | 'balanced' | 'fast' | 'custom';

export interface SchedulePreferences {
  targetMultiplier: number;
  maxMultiplier: number;
  expiryHours: number;
}

export interface Preferences {
  strategy: StrategyProfile;
  defaultTxType: TransactionType;
  urgency: number;
  schedule: SchedulePreferences;
  showAdvancedFields: boolean;
}

interface PreferencesContextType {
  preferences: Preferences;
  setStrategy: (strategy: StrategyProfile) => void;
  updatePreferences: (updates: Partial<Preferences>) => void;
}

const STORAGE_KEY = 'gweizy_preferences_v1';

const STRATEGY_PRESETS: Record<Exclude<StrategyProfile, 'custom'>, Pick<Preferences, 'urgency' | 'schedule'>> = {
  saver: {
    urgency: 0.3,
    schedule: {
      targetMultiplier: 0.8,
      maxMultiplier: 1.05,
      expiryHours: 48
    }
  },
  balanced: {
    urgency: 0.5,
    schedule: {
      targetMultiplier: 0.85,
      maxMultiplier: 1.1,
      expiryHours: 24
    }
  },
  fast: {
    urgency: 0.8,
    schedule: {
      targetMultiplier: 0.95,
      maxMultiplier: 1.2,
      expiryHours: 6
    }
  }
};

const DEFAULT_PREFERENCES: Preferences = {
  strategy: 'balanced',
  defaultTxType: 'swap',
  urgency: STRATEGY_PRESETS.balanced.urgency,
  schedule: STRATEGY_PRESETS.balanced.schedule,
  showAdvancedFields: false
};

const PreferencesContext = createContext<PreferencesContextType | undefined>(undefined);

export const PreferencesProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [preferences, setPreferences] = useState<Preferences>(() => {
    if (typeof window === 'undefined') return DEFAULT_PREFERENCES;
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return DEFAULT_PREFERENCES;
    try {
      const parsed = JSON.parse(stored) as Preferences;
      return {
        ...DEFAULT_PREFERENCES,
        ...parsed,
        schedule: {
          ...DEFAULT_PREFERENCES.schedule,
          ...parsed.schedule
        }
      };
    } catch {
      return DEFAULT_PREFERENCES;
    }
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  }, [preferences]);

  const setStrategy = (strategy: StrategyProfile) => {
    if (strategy === 'custom') {
      setPreferences((prev) => ({ ...prev, strategy }));
      return;
    }
    const preset = STRATEGY_PRESETS[strategy];
    setPreferences((prev) => ({
      ...prev,
      strategy,
      urgency: preset.urgency,
      schedule: { ...prev.schedule, ...preset.schedule }
    }));
  };

  const updatePreferences = (updates: Partial<Preferences>) => {
    setPreferences((prev) => ({
      ...prev,
      ...updates,
      schedule: {
        ...prev.schedule,
        ...(updates.schedule || {})
      }
    }));
  };

  const value = useMemo(
    () => ({
      preferences,
      setStrategy,
      updatePreferences
    }),
    [preferences]
  );

  return (
    <PreferencesContext.Provider value={value}>
      {children}
    </PreferencesContext.Provider>
  );
};

export const usePreferences = (): PreferencesContextType => {
  const context = useContext(PreferencesContext);
  if (!context) {
    throw new Error('usePreferences must be used within a PreferencesProvider');
  }
  return context;
};
