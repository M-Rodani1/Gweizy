type OriginResolver = () => string | null;

const defaultOriginResolver: OriginResolver = () => {
  if (typeof globalThis === 'undefined') {
    return null;
  }

  const location = (globalThis as { location?: Location }).location;
  if (!location?.origin) {
    return null;
  }

  return location.origin;
};

function isSafeRelativePath(value: string): boolean {
  if (!value.startsWith('/')) {
    return false;
  }

  if (value.startsWith('//') || value.startsWith('/\\')) {
    return false;
  }

  return true;
}

export function sanitizeRedirectUrl(
  target: string,
  allowedOrigins?: string[],
  fallback: string = '/'
): string {
  if (typeof target !== 'string') {
    return fallback;
  }

  const trimmed = target.trim();
  if (trimmed === '') {
    return fallback;
  }

  if (trimmed.startsWith('#') || trimmed.startsWith('?')) {
    return trimmed;
  }

  if (isSafeRelativePath(trimmed)) {
    return trimmed;
  }

  const origin = defaultOriginResolver();
  const origins = allowedOrigins ?? (origin ? [origin] : []);
  if (origins.length === 0) {
    return fallback;
  }

  try {
    const url = origin ? new URL(trimmed, origin) : new URL(trimmed);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return fallback;
    }

    if (!origins.includes(url.origin)) {
      return fallback;
    }

    return url.toString();
  } catch {
    return fallback;
  }
}
