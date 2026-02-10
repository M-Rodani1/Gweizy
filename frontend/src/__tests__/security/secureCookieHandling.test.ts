import { describe, it, expect, beforeEach } from 'vitest';
import { buildCookie } from '../../utils/cookies';

describe('secure cookie handling', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'location', {
      value: { protocol: 'https:' },
      writable: true,
    });
  });

  it('builds secure cookies with SameSite and Path', () => {
    const cookie = buildCookie('session', 'token');
    expect(cookie).toContain('session=token');
    expect(cookie).toContain('Path=/');
    expect(cookie).toContain('SameSite=Strict');
    expect(cookie).toContain('Secure');
  });

  it('respects custom cookie options', () => {
    const cookie = buildCookie('prefs', '1', {
      maxAgeSeconds: 3600,
      sameSite: 'Lax',
      path: '/app',
      secure: false,
    });
    expect(cookie).toContain('Max-Age=3600');
    expect(cookie).toContain('SameSite=Lax');
    expect(cookie).toContain('Path=/app');
    expect(cookie).not.toContain('Secure');
  });
});
