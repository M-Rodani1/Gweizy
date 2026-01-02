/**
 * Feature flags system
 * Allows enabling/disabling features without code changes
 */

import { FEATURE_FLAGS } from '../constants';

class FeatureFlagManager {
  private flags: Record<string, boolean> = { ...FEATURE_FLAGS };

  /**
   * Check if a feature is enabled
   */
  isEnabled(flag: keyof typeof FEATURE_FLAGS): boolean {
    // Check localStorage for overrides
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(`feature_${flag}`);
      if (stored !== null) {
        return stored === 'true';
      }
    }
    
    return this.flags[flag] ?? false;
  }

  /**
   * Enable a feature
   */
  enable(flag: keyof typeof FEATURE_FLAGS): void {
    this.flags[flag] = true;
    if (typeof window !== 'undefined') {
      localStorage.setItem(`feature_${flag}`, 'true');
    }
  }

  /**
   * Disable a feature
   */
  disable(flag: keyof typeof FEATURE_FLAGS): void {
    this.flags[flag] = false;
    if (typeof window !== 'undefined') {
      localStorage.setItem(`feature_${flag}`, 'false');
    }
  }

  /**
   * Get all feature flags
   */
  getAllFlags(): Record<string, boolean> {
    return { ...this.flags };
  }
}

export const featureFlags = new FeatureFlagManager();
