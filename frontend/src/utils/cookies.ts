/**
 * Secure cookie helpers.
 * Note: HttpOnly cannot be set from client-side JavaScript.
 */

export interface CookieOptions {
  maxAgeSeconds?: number;
  path?: string;
  sameSite?: 'Strict' | 'Lax' | 'None';
  secure?: boolean;
}

export function buildCookie(
  name: string,
  value: string,
  options: CookieOptions = {}
): string {
  const parts = [`${encodeURIComponent(name)}=${encodeURIComponent(value)}`];

  if (options.maxAgeSeconds !== undefined) {
    parts.push(`Max-Age=${options.maxAgeSeconds}`);
  }

  parts.push(`Path=${options.path ?? '/'}`);

  const sameSite = options.sameSite ?? 'Strict';
  parts.push(`SameSite=${sameSite}`);

  const secure = options.secure ?? (typeof location !== 'undefined' && location.protocol === 'https:');
  if (secure) {
    parts.push('Secure');
  }

  return parts.join('; ');
}

export function setSecureCookie(name: string, value: string, options: CookieOptions = {}): void {
  const cookie = buildCookie(name, value, options);
  document.cookie = cookie;
}

export function clearCookie(name: string, options: CookieOptions = {}): void {
  const expiredOptions: CookieOptions = {
    ...options,
    maxAgeSeconds: 0,
  };
  document.cookie = buildCookie(name, '', expiredOptions);
}
