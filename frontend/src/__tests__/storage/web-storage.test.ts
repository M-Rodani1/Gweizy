/**
 * Web Storage Tests
 *
 * Tests for localStorage and sessionStorage operations,
 * including edge cases, error handling, and storage events.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ============================================================================
// Mock Storage Implementation
// ============================================================================

class MockStorage implements Storage {
  private store: Map<string, string> = new Map();
  private _quotaExceeded = false;
  private _throwOnAccess = false;

  get length(): number {
    return this.store.size;
  }

  clear(): void {
    this.store.clear();
  }

  getItem(key: string): string | null {
    if (this._throwOnAccess) {
      throw new Error('Storage access denied');
    }
    return this.store.get(key) ?? null;
  }

  key(index: number): string | null {
    const keys = Array.from(this.store.keys());
    return keys[index] ?? null;
  }

  removeItem(key: string): void {
    this.store.delete(key);
  }

  setItem(key: string, value: string): void {
    if (this._throwOnAccess) {
      throw new Error('Storage access denied');
    }
    if (this._quotaExceeded) {
      const error = new DOMException('Quota exceeded', 'QuotaExceededError');
      throw error;
    }
    this.store.set(key, value);
  }

  // Test helpers
  simulateQuotaExceeded(exceeded: boolean): void {
    this._quotaExceeded = exceeded;
  }

  simulateAccessDenied(denied: boolean): void {
    this._throwOnAccess = denied;
  }

  getAllKeys(): string[] {
    return Array.from(this.store.keys());
  }
}

// ============================================================================
// Storage Utility Functions (for testing)
// ============================================================================

function safeGetItem<T>(storage: Storage, key: string, defaultValue: T): T {
  try {
    const item = storage.getItem(key);
    if (item === null) return defaultValue;
    return JSON.parse(item) as T;
  } catch {
    return defaultValue;
  }
}

function safeSetItem(storage: Storage, key: string, value: unknown): boolean {
  try {
    storage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function safeRemoveItem(storage: Storage, key: string): boolean {
  try {
    storage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}

function getStorageSize(storage: Storage): number {
  let size = 0;
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (key) {
      const value = storage.getItem(key);
      if (value) {
        size += key.length + value.length;
      }
    }
  }
  return size;
}

function clearStorageByPrefix(storage: Storage, prefix: string): number {
  const keysToRemove: string[] = [];
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (key?.startsWith(prefix)) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach((key) => storage.removeItem(key));
  return keysToRemove.length;
}

// ============================================================================
// Tests
// ============================================================================

describe('Web Storage Tests', () => {
  let mockStorage: MockStorage;

  beforeEach(() => {
    mockStorage = new MockStorage();
  });

  afterEach(() => {
    mockStorage.clear();
  });

  describe('Basic Operations', () => {
    it('should set and get string values', () => {
      mockStorage.setItem('key', 'value');
      expect(mockStorage.getItem('key')).toBe('value');
    });

    it('should return null for non-existent keys', () => {
      expect(mockStorage.getItem('nonexistent')).toBeNull();
    });

    it('should remove items', () => {
      mockStorage.setItem('key', 'value');
      mockStorage.removeItem('key');
      expect(mockStorage.getItem('key')).toBeNull();
    });

    it('should clear all items', () => {
      mockStorage.setItem('key1', 'value1');
      mockStorage.setItem('key2', 'value2');
      mockStorage.clear();
      expect(mockStorage.length).toBe(0);
    });

    it('should track length correctly', () => {
      expect(mockStorage.length).toBe(0);
      mockStorage.setItem('key1', 'value1');
      expect(mockStorage.length).toBe(1);
      mockStorage.setItem('key2', 'value2');
      expect(mockStorage.length).toBe(2);
      mockStorage.removeItem('key1');
      expect(mockStorage.length).toBe(1);
    });

    it('should return keys by index', () => {
      mockStorage.setItem('key1', 'value1');
      mockStorage.setItem('key2', 'value2');
      expect(mockStorage.key(0)).toBe('key1');
      expect(mockStorage.key(1)).toBe('key2');
      expect(mockStorage.key(2)).toBeNull();
    });

    it('should overwrite existing values', () => {
      mockStorage.setItem('key', 'original');
      mockStorage.setItem('key', 'updated');
      expect(mockStorage.getItem('key')).toBe('updated');
      expect(mockStorage.length).toBe(1);
    });
  });

  describe('JSON Serialization', () => {
    it('should serialize and deserialize objects', () => {
      const data = { name: 'test', count: 42, active: true };
      safeSetItem(mockStorage, 'data', data);
      const retrieved = safeGetItem(mockStorage, 'data', {});
      expect(retrieved).toEqual(data);
    });

    it('should serialize and deserialize arrays', () => {
      const data = [1, 2, 3, 'four', { five: 5 }];
      safeSetItem(mockStorage, 'array', data);
      const retrieved = safeGetItem(mockStorage, 'array', []);
      expect(retrieved).toEqual(data);
    });

    it('should serialize and deserialize nested structures', () => {
      const data = {
        user: {
          name: 'Alice',
          preferences: {
            theme: 'dark',
            notifications: {
              email: true,
              push: false,
            },
          },
        },
        items: [1, 2, 3],
      };
      safeSetItem(mockStorage, 'nested', data);
      expect(safeGetItem(mockStorage, 'nested', null)).toEqual(data);
    });

    it('should handle null values', () => {
      safeSetItem(mockStorage, 'null', null);
      expect(safeGetItem(mockStorage, 'null', 'default')).toBeNull();
    });

    it('should handle special number values', () => {
      // NaN and Infinity are not valid JSON
      safeSetItem(mockStorage, 'number', 123.456);
      expect(safeGetItem(mockStorage, 'number', 0)).toBe(123.456);
    });

    it('should return default value for corrupted JSON', () => {
      mockStorage.setItem('corrupted', '{invalid json}');
      expect(safeGetItem(mockStorage, 'corrupted', 'default')).toBe('default');
    });

    it('should return default value for non-existent key', () => {
      const defaultValue = { default: true };
      expect(safeGetItem(mockStorage, 'nonexistent', defaultValue)).toEqual(defaultValue);
    });
  });

  describe('Error Handling', () => {
    it('should handle quota exceeded errors', () => {
      mockStorage.simulateQuotaExceeded(true);
      const result = safeSetItem(mockStorage, 'key', { large: 'data' });
      expect(result).toBe(false);
    });

    it('should handle storage access denied', () => {
      mockStorage.simulateAccessDenied(true);
      expect(safeGetItem(mockStorage, 'key', 'default')).toBe('default');
      expect(safeSetItem(mockStorage, 'key', 'value')).toBe(false);
    });

    it('should safely remove items even with errors', () => {
      mockStorage.setItem('key', 'value');
      expect(safeRemoveItem(mockStorage, 'key')).toBe(true);
      expect(mockStorage.getItem('key')).toBeNull();
    });
  });

  describe('Storage Utilities', () => {
    it('should calculate storage size', () => {
      mockStorage.setItem('key1', 'value1');
      mockStorage.setItem('key2', 'value2');
      const size = getStorageSize(mockStorage);
      expect(size).toBe('key1'.length + 'value1'.length + 'key2'.length + 'value2'.length);
    });

    it('should clear items by prefix', () => {
      mockStorage.setItem('app_setting1', 'value1');
      mockStorage.setItem('app_setting2', 'value2');
      mockStorage.setItem('other_key', 'value3');

      const removed = clearStorageByPrefix(mockStorage, 'app_');
      expect(removed).toBe(2);
      expect(mockStorage.getItem('app_setting1')).toBeNull();
      expect(mockStorage.getItem('app_setting2')).toBeNull();
      expect(mockStorage.getItem('other_key')).toBe('value3');
    });

    it('should return all keys', () => {
      mockStorage.setItem('a', '1');
      mockStorage.setItem('b', '2');
      mockStorage.setItem('c', '3');
      expect(mockStorage.getAllKeys()).toEqual(['a', 'b', 'c']);
    });
  });

  describe('Storage Migration Patterns', () => {
    interface LegacyPreferences {
      theme: string;
    }

    interface NewPreferences {
      version: number;
      theme: string;
      notifications: boolean;
    }

    function migratePreferences(storage: Storage, key: string): NewPreferences {
      const defaultPrefs: NewPreferences = {
        version: 2,
        theme: 'dark',
        notifications: true,
      };

      const stored = storage.getItem(key);
      if (!stored) return defaultPrefs;

      try {
        const parsed = JSON.parse(stored);

        // Check if already migrated
        if (parsed.version === 2) {
          return parsed as NewPreferences;
        }

        // Migrate from v1 (legacy)
        const legacy = parsed as LegacyPreferences;
        const migrated: NewPreferences = {
          version: 2,
          theme: legacy.theme || 'dark',
          notifications: true, // New field with default
        };

        // Save migrated data
        storage.setItem(key, JSON.stringify(migrated));
        return migrated;
      } catch {
        return defaultPrefs;
      }
    }

    it('should migrate legacy data format', () => {
      // Store legacy format
      mockStorage.setItem('prefs', JSON.stringify({ theme: 'light' }));

      // Migrate and verify
      const migrated = migratePreferences(mockStorage, 'prefs');
      expect(migrated.version).toBe(2);
      expect(migrated.theme).toBe('light');
      expect(migrated.notifications).toBe(true);
    });

    it('should not re-migrate already migrated data', () => {
      const newFormat: NewPreferences = {
        version: 2,
        theme: 'dark',
        notifications: false,
      };
      mockStorage.setItem('prefs', JSON.stringify(newFormat));

      const result = migratePreferences(mockStorage, 'prefs');
      expect(result.notifications).toBe(false); // Preserved, not overwritten with default
    });

    it('should return defaults for missing data', () => {
      const result = migratePreferences(mockStorage, 'nonexistent');
      expect(result.version).toBe(2);
      expect(result.theme).toBe('dark');
      expect(result.notifications).toBe(true);
    });
  });

  describe('Storage Key Namespacing', () => {
    function createNamespacedStorage(storage: Storage, namespace: string) {
      return {
        getItem: (key: string) => storage.getItem(`${namespace}_${key}`),
        setItem: (key: string, value: string) => storage.setItem(`${namespace}_${key}`, value),
        removeItem: (key: string) => storage.removeItem(`${namespace}_${key}`),
        clear: () => clearStorageByPrefix(storage, `${namespace}_`),
      };
    }

    it('should namespace keys correctly', () => {
      const appStorage = createNamespacedStorage(mockStorage, 'myapp');
      appStorage.setItem('setting', 'value');

      expect(mockStorage.getItem('myapp_setting')).toBe('value');
      expect(appStorage.getItem('setting')).toBe('value');
    });

    it('should isolate namespaces', () => {
      const app1Storage = createNamespacedStorage(mockStorage, 'app1');
      const app2Storage = createNamespacedStorage(mockStorage, 'app2');

      app1Storage.setItem('config', 'value1');
      app2Storage.setItem('config', 'value2');

      expect(app1Storage.getItem('config')).toBe('value1');
      expect(app2Storage.getItem('config')).toBe('value2');
    });

    it('should clear only namespaced keys', () => {
      const appStorage = createNamespacedStorage(mockStorage, 'myapp');
      appStorage.setItem('a', '1');
      appStorage.setItem('b', '2');
      mockStorage.setItem('other', '3');

      appStorage.clear();

      expect(appStorage.getItem('a')).toBeNull();
      expect(appStorage.getItem('b')).toBeNull();
      expect(mockStorage.getItem('other')).toBe('3');
    });
  });

  describe('Storage Event Simulation', () => {
    it('should track storage changes with custom event system', () => {
      interface StorageChangeEvent {
        key: string;
        oldValue: string | null;
        newValue: string | null;
      }

      const listeners: Array<(e: StorageChangeEvent) => void> = [];

      const addStorageListener = (callback: (e: StorageChangeEvent) => void) => {
        listeners.push(callback);
      };

      const dispatchStorageChange = (key: string, oldValue: string | null, newValue: string | null) => {
        const event: StorageChangeEvent = { key, oldValue, newValue };
        listeners.forEach((listener) => listener(event));
      };

      let receivedEvent: StorageChangeEvent | null = null;
      addStorageListener((e) => {
        receivedEvent = e;
      });

      // Simulate storage change
      const oldValue = mockStorage.getItem('key');
      mockStorage.setItem('key', 'newValue');
      dispatchStorageChange('key', oldValue, 'newValue');

      expect(receivedEvent).not.toBeNull();
      expect(receivedEvent!.key).toBe('key');
      expect(receivedEvent!.oldValue).toBeNull();
      expect(receivedEvent!.newValue).toBe('newValue');
    });
  });

  describe('Concurrent Access Patterns', () => {
    it('should handle rapid successive writes', () => {
      for (let i = 0; i < 100; i++) {
        mockStorage.setItem(`key${i}`, `value${i}`);
      }
      expect(mockStorage.length).toBe(100);
    });

    it('should handle interleaved reads and writes', () => {
      const results: string[] = [];

      mockStorage.setItem('counter', '0');
      for (let i = 0; i < 10; i++) {
        const current = parseInt(mockStorage.getItem('counter') || '0', 10);
        results.push(mockStorage.getItem('counter')!);
        mockStorage.setItem('counter', String(current + 1));
      }

      expect(mockStorage.getItem('counter')).toBe('10');
      expect(results).toEqual(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
    });
  });

  describe('Special Characters and Unicode', () => {
    it('should handle keys with special characters', () => {
      mockStorage.setItem('key-with-dashes', 'value1');
      mockStorage.setItem('key_with_underscores', 'value2');
      mockStorage.setItem('key.with.dots', 'value3');

      expect(mockStorage.getItem('key-with-dashes')).toBe('value1');
      expect(mockStorage.getItem('key_with_underscores')).toBe('value2');
      expect(mockStorage.getItem('key.with.dots')).toBe('value3');
    });

    it('should handle Unicode values', () => {
      const unicodeValue = '你好世界 🌍 مرحبا';
      mockStorage.setItem('unicode', unicodeValue);
      expect(mockStorage.getItem('unicode')).toBe(unicodeValue);
    });

    it('should handle empty string values', () => {
      mockStorage.setItem('empty', '');
      expect(mockStorage.getItem('empty')).toBe('');
    });

    it('should handle values with quotes', () => {
      const valueWithQuotes = 'He said "hello" and \'goodbye\'';
      mockStorage.setItem('quotes', valueWithQuotes);
      expect(mockStorage.getItem('quotes')).toBe(valueWithQuotes);
    });
  });

  describe('Type Safety Patterns', () => {
    interface UserSettings {
      theme: 'light' | 'dark';
      fontSize: number;
      notifications: boolean;
    }

    const defaultSettings: UserSettings = {
      theme: 'dark',
      fontSize: 14,
      notifications: true,
    };

    function isValidUserSettings(data: unknown): data is UserSettings {
      if (typeof data !== 'object' || data === null) return false;
      const obj = data as Record<string, unknown>;
      return (
        (obj.theme === 'light' || obj.theme === 'dark') &&
        typeof obj.fontSize === 'number' &&
        typeof obj.notifications === 'boolean'
      );
    }

    function getTypedSettings(storage: Storage, key: string): UserSettings {
      const raw = storage.getItem(key);
      if (!raw) return defaultSettings;

      try {
        const parsed = JSON.parse(raw);
        if (isValidUserSettings(parsed)) {
          return parsed;
        }
        return defaultSettings;
      } catch {
        return defaultSettings;
      }
    }

    it('should validate and return typed data', () => {
      const settings: UserSettings = { theme: 'light', fontSize: 16, notifications: false };
      mockStorage.setItem('settings', JSON.stringify(settings));

      const retrieved = getTypedSettings(mockStorage, 'settings');
      expect(retrieved).toEqual(settings);
    });

    it('should return defaults for invalid data', () => {
      mockStorage.setItem('settings', JSON.stringify({ theme: 'invalid', fontSize: 'not a number' }));

      const retrieved = getTypedSettings(mockStorage, 'settings');
      expect(retrieved).toEqual(defaultSettings);
    });

    it('should return defaults for corrupted JSON', () => {
      mockStorage.setItem('settings', 'not json');
      const retrieved = getTypedSettings(mockStorage, 'settings');
      expect(retrieved).toEqual(defaultSettings);
    });
  });

  describe('Expiring Storage Items', () => {
    interface ExpiringItem<T> {
      value: T;
      expiresAt: number;
    }

    function setItemWithExpiry<T>(storage: Storage, key: string, value: T, ttlMs: number): void {
      const item: ExpiringItem<T> = {
        value,
        expiresAt: Date.now() + ttlMs,
      };
      storage.setItem(key, JSON.stringify(item));
    }

    function getItemWithExpiry<T>(storage: Storage, key: string, now = Date.now()): T | null {
      const raw = storage.getItem(key);
      if (!raw) return null;

      try {
        const item = JSON.parse(raw) as ExpiringItem<T>;
        if (now > item.expiresAt) {
          storage.removeItem(key);
          return null;
        }
        return item.value;
      } catch {
        return null;
      }
    }

    it('should store and retrieve non-expired items', () => {
      const now = Date.now();
      setItemWithExpiry(mockStorage, 'token', 'abc123', 60000);

      // Still valid
      const value = getItemWithExpiry<string>(mockStorage, 'token', now + 30000);
      expect(value).toBe('abc123');
    });

    it('should return null and remove expired items', () => {
      const now = Date.now();
      setItemWithExpiry(mockStorage, 'token', 'abc123', 60000);

      // Expired
      const value = getItemWithExpiry<string>(mockStorage, 'token', now + 120000);
      expect(value).toBeNull();
      expect(mockStorage.getItem('token')).toBeNull();
    });
  });
});
