export type FeatureFlag = {
  key: string;
  description?: string;
  enabled: boolean;
};

const STORAGE_KEY = 'gweizy_feature_flags';

let registry = new Map<string, FeatureFlag>();

export const registerFeatureFlags = (flags: FeatureFlag[]) => {
  registry = new Map(flags.map((flag) => [flag.key, flag]));
  loadFeatureFlags();
};

export const isFeatureEnabled = (key: string): boolean => {
  return registry.get(key)?.enabled ?? false;
};

export const setFeatureEnabled = (key: string, enabled: boolean) => {
  const existing = registry.get(key);
  if (!existing) {
    registry.set(key, { key, enabled });
  } else {
    registry.set(key, { ...existing, enabled });
  }
  saveFeatureFlags();
};

export const getFeatureFlags = (): FeatureFlag[] => Array.from(registry.values());

const loadFeatureFlags = () => {
  if (typeof window === 'undefined') return;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const stored = JSON.parse(raw) as Record<string, boolean>;
    Object.entries(stored).forEach(([key, enabled]) => {
      const existing = registry.get(key);
      if (existing) {
        registry.set(key, { ...existing, enabled });
      } else {
        registry.set(key, { key, enabled });
      }
    });
  } catch {
    // ignore storage errors
  }
};

const saveFeatureFlags = () => {
  if (typeof window === 'undefined') return;
  try {
    const snapshot: Record<string, boolean> = {};
    registry.forEach((flag) => {
      snapshot[flag.key] = flag.enabled;
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  } catch {
    // ignore storage errors
  }
};
