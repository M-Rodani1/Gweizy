import { secureStorage } from './secureStorage';

const SESSION_KEY = 'session';

export interface SessionInfo {
  token: string;
  createdAt: number;
  updatedAt: number;
  expiresAt: number;
}

export async function startSession(token: string, ttlMs = 30 * 60 * 1000): Promise<SessionInfo> {
  const now = Date.now();
  const session: SessionInfo = {
    token,
    createdAt: now,
    updatedAt: now,
    expiresAt: now + ttlMs,
  };
  await secureStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
}

export async function getSession(): Promise<SessionInfo | null> {
  const raw = await secureStorage.getItem(SESSION_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as SessionInfo;
    if (!parsed.token || !parsed.expiresAt) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export async function isSessionValid(): Promise<boolean> {
  const session = await getSession();
  if (!session) return false;
  return Date.now() < session.expiresAt;
}

export async function touchSession(ttlMs = 30 * 60 * 1000): Promise<SessionInfo | null> {
  const session = await getSession();
  if (!session) return null;
  const now = Date.now();
  const updated: SessionInfo = {
    ...session,
    updatedAt: now,
    expiresAt: now + ttlMs,
  };
  await secureStorage.setItem(SESSION_KEY, JSON.stringify(updated));
  return updated;
}

export async function updateSessionToken(token: string): Promise<SessionInfo | null> {
  const session = await getSession();
  if (!session) return null;
  const now = Date.now();
  const updated: SessionInfo = {
    ...session,
    token,
    updatedAt: now,
  };
  await secureStorage.setItem(SESSION_KEY, JSON.stringify(updated));
  return updated;
}

export function clearSession(): void {
  secureStorage.removeItem(SESSION_KEY);
}
