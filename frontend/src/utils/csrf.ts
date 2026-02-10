/**
 * CSRF token handling utilities.
 */

const META_NAMES = ['csrf-token', 'xsrf-token'];
const COOKIE_NAMES = ['csrf_token', 'xsrf_token', 'XSRF-TOKEN'];

function readMetaToken(doc: Document): string | null {
  for (const name of META_NAMES) {
    const meta = doc.querySelector(`meta[name="${name}"]`);
    const content = meta?.getAttribute('content');
    if (content) {
      return content;
    }
  }
  return null;
}

function parseCookies(cookieString: string): Record<string, string> {
  const result: Record<string, string> = {};
  const parts = cookieString.split(';');
  for (const part of parts) {
    const [rawKey, ...rest] = part.split('=');
    if (!rawKey || rest.length === 0) continue;
    const key = rawKey.trim();
    const value = rest.join('=').trim();
    if (key) {
      result[key] = decodeURIComponent(value);
    }
  }
  return result;
}

export function getCsrfToken(): string | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const metaToken = readMetaToken(document);
  if (metaToken) {
    return metaToken;
  }

  const cookies = parseCookies(document.cookie ?? '');
  for (const name of COOKIE_NAMES) {
    if (cookies[name]) {
      return cookies[name];
    }
  }

  return null;
}

export function buildCsrfHeaders(token: string | null): Record<string, string> {
  if (!token) {
    return {};
  }

  return {
    'X-CSRF-Token': token,
  };
}
