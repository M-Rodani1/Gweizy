/**
 * Safe localStorage helpers.
 * Prevents hard crashes when storage is unavailable or blocked.
 */

export function safeGetLocalStorageItem(key: string): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function safeSetLocalStorageItem(key: string, value: string): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore storage failures to avoid breaking app startup.
  }
}

export function safeRemoveLocalStorageItem(key: string): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(key);
  } catch {
    // Ignore storage failures to avoid breaking app startup.
  }
}
