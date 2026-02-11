import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  registerFeatureFlags,
  isFeatureEnabled,
  setFeatureEnabled,
  getFeatureFlags
} from '../../utils/featureFlags';

describe('feature flags', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn()
    });
  });

  it('registers flags and checks enabled state', () => {
    registerFeatureFlags([
      { key: 'new-ui', enabled: false },
      { key: 'beta-flow', enabled: true }
    ]);

    expect(isFeatureEnabled('new-ui')).toBe(false);
    expect(isFeatureEnabled('beta-flow')).toBe(true);
  });

  it('updates flags and persists to storage', () => {
    registerFeatureFlags([{ key: 'fast-path', enabled: false }]);
    setFeatureEnabled('fast-path', true);

    const flags = getFeatureFlags();
    expect(flags.find((flag) => flag.key === 'fast-path')?.enabled).toBe(true);
    expect(localStorage.setItem).toHaveBeenCalled();
  });
});
