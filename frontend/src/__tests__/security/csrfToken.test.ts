import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { getCsrfToken, buildCsrfHeaders } from '../../utils/csrf';

describe('CSRF token handling', () => {
  const originalHead = document.head.innerHTML;
  const originalCookie = document.cookie;

  beforeEach(() => {
    document.head.innerHTML = '';
    document.cookie = '';
  });

  afterEach(() => {
    document.head.innerHTML = originalHead;
    document.cookie = originalCookie;
  });

  it('prefers meta tag token when present', () => {
    const meta = document.createElement('meta');
    meta.setAttribute('name', 'csrf-token');
    meta.setAttribute('content', 'meta-token');
    document.head.appendChild(meta);

    document.cookie = 'csrf_token=cookie-token';

    expect(getCsrfToken()).toBe('meta-token');
  });

  it('falls back to cookie token', () => {
    document.cookie = 'csrf_token=cookie-token';
    expect(getCsrfToken()).toBe('cookie-token');
  });

  it('builds CSRF headers when token exists', () => {
    const headers = buildCsrfHeaders('token-123');
    expect(headers).toEqual({ 'X-CSRF-Token': 'token-123' });
  });

  it('returns empty headers when token is missing', () => {
    const headers = buildCsrfHeaders(null);
    expect(headers).toEqual({});
  });
});
