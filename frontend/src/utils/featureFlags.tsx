/**
 * Feature flags system for controlled feature rollouts.
 *
 * Allows enabling/disabling features without code changes.
 * Supports localStorage persistence and URL parameter overrides.
 *
 * @module utils/featureFlags
 */

import { createContext, useContext, useState, useMemo, useEffect } from 'react';
import type { ReactNode } from 'react';
import { FEATURE_FLAGS } from '../constants';

type FeatureFlagKey = keyof typeof FEATURE_FLAGS;

const STORAGE_PREFIX = 'feature_';

/**
 * Feature flag manager class for non-React usage.
 */
class FeatureFlagManager {
  private flags: Record<string, boolean> = { ...FEATURE_FLAGS };
  private listeners: Set<() => void> = new Set();

  /**
   * Check if a feature is enabled
   */
  isEnabled(flag: FeatureFlagKey): boolean {
    // Check URL params first (highest priority)
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const urlValue = params.get(`ff_${flag}`);
      if (urlValue !== null) {
        return urlValue === 'true' || urlValue === '1';
      }

      // Check localStorage for overrides
      const stored = localStorage.getItem(`${STORAGE_PREFIX}${flag}`);
      if (stored !== null) {
        return stored === 'true';
      }
    }

    return this.flags[flag] ?? false;
  }

  /**
   * Enable a feature
   */
  enable(flag: FeatureFlagKey): void {
    this.flags[flag] = true;
    if (typeof window !== 'undefined') {
      localStorage.setItem(`${STORAGE_PREFIX}${flag}`, 'true');
    }
    this.notify();
  }

  /**
   * Disable a feature
   */
  disable(flag: FeatureFlagKey): void {
    this.flags[flag] = false;
    if (typeof window !== 'undefined') {
      localStorage.setItem(`${STORAGE_PREFIX}${flag}`, 'false');
    }
    this.notify();
  }

  /**
   * Toggle a feature
   */
  toggle(flag: FeatureFlagKey): boolean {
    const newValue = !this.isEnabled(flag);
    if (newValue) {
      this.enable(flag);
    } else {
      this.disable(flag);
    }
    return newValue;
  }

  /**
   * Reset a feature to its default value
   */
  reset(flag: FeatureFlagKey): void {
    this.flags[flag] = FEATURE_FLAGS[flag];
    if (typeof window !== 'undefined') {
      localStorage.removeItem(`${STORAGE_PREFIX}${flag}`);
    }
    this.notify();
  }

  /**
   * Reset all features to defaults
   */
  resetAll(): void {
    this.flags = { ...FEATURE_FLAGS };
    if (typeof window !== 'undefined') {
      Object.keys(FEATURE_FLAGS).forEach((flag) => {
        localStorage.removeItem(`${STORAGE_PREFIX}${flag}`);
      });
    }
    this.notify();
  }

  /**
   * Get all feature flags with current values
   */
  getAllFlags(): Record<FeatureFlagKey, boolean> {
    const result: Record<string, boolean> = {};
    for (const key of Object.keys(FEATURE_FLAGS) as FeatureFlagKey[]) {
      result[key] = this.isEnabled(key);
    }
    return result as Record<FeatureFlagKey, boolean>;
  }

  /**
   * Subscribe to flag changes
   */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    this.listeners.forEach((listener) => listener());
  }
}

export const featureFlags = new FeatureFlagManager();

// ========================================
// React Integration
// ========================================

interface FeatureFlagContextValue {
  isEnabled: (flag: FeatureFlagKey) => boolean;
  enable: (flag: FeatureFlagKey) => void;
  disable: (flag: FeatureFlagKey) => void;
  toggle: (flag: FeatureFlagKey) => boolean;
  reset: (flag: FeatureFlagKey) => void;
  resetAll: () => void;
  flags: Record<FeatureFlagKey, boolean>;
}

const FeatureFlagContext = createContext<FeatureFlagContextValue | null>(null);

/**
 * Provider for React feature flag integration.
 *
 * @example
 * ```tsx
 * <FeatureFlagProvider>
 *   <App />
 * </FeatureFlagProvider>
 * ```
 */
export function FeatureFlagProvider({ children }: { children: ReactNode }) {
  const [, forceUpdate] = useState({});

  useEffect(() => {
    return featureFlags.subscribe(() => forceUpdate({}));
  }, []);

  const value = useMemo<FeatureFlagContextValue>(
    () => ({
      isEnabled: (flag) => featureFlags.isEnabled(flag),
      enable: (flag) => featureFlags.enable(flag),
      disable: (flag) => featureFlags.disable(flag),
      toggle: (flag) => featureFlags.toggle(flag),
      reset: (flag) => featureFlags.reset(flag),
      resetAll: () => featureFlags.resetAll(),
      flags: featureFlags.getAllFlags(),
    }),
    []
  );

  return (
    <FeatureFlagContext.Provider value={value}>
      {children}
    </FeatureFlagContext.Provider>
  );
}

/**
 * Hook to access feature flags.
 */
export function useFeatureFlags(): FeatureFlagContextValue {
  const context = useContext(FeatureFlagContext);
  if (!context) {
    return {
      isEnabled: (flag) => featureFlags.isEnabled(flag),
      enable: (flag) => featureFlags.enable(flag),
      disable: (flag) => featureFlags.disable(flag),
      toggle: (flag) => featureFlags.toggle(flag),
      reset: (flag) => featureFlags.reset(flag),
      resetAll: () => featureFlags.resetAll(),
      flags: featureFlags.getAllFlags(),
    };
  }
  return context;
}

/**
 * Hook to check a single feature flag.
 */
export function useFeatureFlag(flag: FeatureFlagKey): boolean {
  const { isEnabled } = useFeatureFlags();
  const [, forceUpdate] = useState({});

  useEffect(() => {
    return featureFlags.subscribe(() => forceUpdate({}));
  }, []);

  return isEnabled(flag);
}

/**
 * Render children only if feature is enabled.
 *
 * @example
 * ```tsx
 * <Feature flag="WEBSOCKET_ENABLED">
 *   <WebSocketComponent />
 * </Feature>
 * ```
 */
export function Feature({
  flag,
  children,
  fallback = null,
}: {
  flag: FeatureFlagKey;
  children: ReactNode;
  fallback?: ReactNode;
}) {
  const enabled = useFeatureFlag(flag);
  return <>{enabled ? children : fallback}</>;
}

export default featureFlags;
