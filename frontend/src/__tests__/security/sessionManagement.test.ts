import { describe, it, expect, vi, beforeEach } from 'vitest';

const store = new Map<string, string>();

vi.mock('../../utils/secureStorage', () => ({
  secureStorage: {
    setItem: vi.fn(async (key: string, value: string) => {
      store.set(key, value);
    }),
    getItem: vi.fn(async (key: string) => store.get(key) ?? null),
    removeItem: vi.fn((key: string) => {
      store.delete(key);
    }),
  },
}));

import {
  startSession,
  getSession,
  isSessionValid,
  touchSession,
  updateSessionToken,
  clearSession,
} from '../../utils/session';

describe('session management', () => {
  beforeEach(() => {
    store.clear();
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2024-01-01T00:00:00Z'));
  });

  it('creates and retrieves a session', async () => {
    await startSession('token-1', 1000);
    const session = await getSession();
    expect(session?.token).toBe('token-1');
  });

  it('validates expiration', async () => {
    await startSession('token-2', 1000);
    expect(await isSessionValid()).toBe(true);
    vi.setSystemTime(new Date('2024-01-01T00:00:02Z'));
    expect(await isSessionValid()).toBe(false);
  });

  it('extends session on touch', async () => {
    await startSession('token-3', 1000);
    vi.setSystemTime(new Date('2024-01-01T00:00:00.500Z'));
    const updated = await touchSession(2000);
    expect(updated?.expiresAt).toBeGreaterThan(Date.now());
  });

  it('updates the session token', async () => {
    await startSession('token-4', 1000);
    const updated = await updateSessionToken('token-5');
    expect(updated?.token).toBe('token-5');
  });

  it('clears the session', async () => {
    await startSession('token-6', 1000);
    clearSession();
    expect(await getSession()).toBeNull();
  });
});
