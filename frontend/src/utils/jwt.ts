const base64UrlDecode = (value: string): string => {
  const normalized = value.replace(/-/g, '+').replace(/_/g, '/');
  const padded = normalized.padEnd(normalized.length + ((4 - (normalized.length % 4)) % 4), '=');
  return atob(padded);
};

export function decodeJwtPayload(token: string): Record<string, any> | null {
  const parts = token.split('.');
  if (parts.length < 2) return null;

  try {
    const json = base64UrlDecode(parts[1]);
    return JSON.parse(json) as Record<string, any>;
  } catch {
    return null;
  }
}

export function getJwtExpiry(token: string): number | null {
  const payload = decodeJwtPayload(token);
  if (!payload || typeof payload.exp !== 'number') {
    return null;
  }
  return payload.exp * 1000;
}

export function isJwtExpired(token: string, now = Date.now()): boolean {
  const expiry = getJwtExpiry(token);
  if (!expiry) return true;
  return now >= expiry;
}

export function shouldRefreshToken(
  token: string,
  now = Date.now(),
  refreshWindowMs = 5 * 60 * 1000
): boolean {
  const expiry = getJwtExpiry(token);
  if (!expiry) return true;
  return expiry - now <= refreshWindowMs;
}
