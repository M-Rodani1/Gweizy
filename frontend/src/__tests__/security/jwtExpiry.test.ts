import { describe, it, expect } from 'vitest';
import { decodeJwtPayload, getJwtExpiry, isJwtExpired, shouldRefreshToken } from '../../utils/jwt';

const base64UrlEncode = (value: string): string =>
  btoa(value).replace(/=/g, '').replace(/\+/g, '-').replace(/\//g, '_');

const buildToken = (payload: Record<string, any>) => {
  const header = base64UrlEncode(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
  const body = base64UrlEncode(JSON.stringify(payload));
  return `${header}.${body}.signature`;
};

describe('JWT expiry monitoring', () => {
  it('decodes payload and extracts exp', () => {
    const exp = Math.floor(Date.now() / 1000) + 60;
    const token = buildToken({ exp });

    expect(decodeJwtPayload(token)?.exp).toBe(exp);
    expect(getJwtExpiry(token)).toBe(exp * 1000);
  });

  it('detects expired tokens', () => {
    const exp = Math.floor(Date.now() / 1000) - 10;
    const token = buildToken({ exp });

    expect(isJwtExpired(token)).toBe(true);
  });

  it('flags tokens nearing expiry for refresh', () => {
    const exp = Math.floor(Date.now() / 1000) + 30;
    const token = buildToken({ exp });

    expect(shouldRefreshToken(token, Date.now(), 60_000)).toBe(true);
  });
});
