export interface SWRCacheOptions {
  maxAgeMs?: number;
  staleWhileRevalidateMs?: number;
}

interface CacheEntry<T> {
  data: T;
  updatedAt: number;
  inFlight?: Promise<T>;
}

const DEFAULT_MAX_AGE = 30_000;
const DEFAULT_STALE = 60_000;

const cache = new Map<string, CacheEntry<unknown>>();

function getConfig(options?: SWRCacheOptions) {
  return {
    maxAgeMs: options?.maxAgeMs ?? DEFAULT_MAX_AGE,
    staleWhileRevalidateMs: options?.staleWhileRevalidateMs ?? DEFAULT_STALE,
  };
}

export async function getWithSWR<T>(
  key: string,
  fetcher: () => Promise<T>,
  options?: SWRCacheOptions
): Promise<T> {
  const now = Date.now();
  const { maxAgeMs, staleWhileRevalidateMs } = getConfig(options);
  const entry = cache.get(key) as CacheEntry<T> | undefined;

  if (entry) {
    const age = now - entry.updatedAt;

    if (age <= maxAgeMs) {
      return entry.data;
    }

    if (age <= maxAgeMs + staleWhileRevalidateMs) {
      if (!entry.inFlight) {
        entry.inFlight = fetcher()
          .then((data) => {
            cache.set(key, { data, updatedAt: Date.now() });
            return data;
          })
          .finally(() => {
            const current = cache.get(key) as CacheEntry<T> | undefined;
            if (current) {
              delete current.inFlight;
            }
          });
      }
      return entry.data;
    }
  }

  if (entry?.inFlight) {
    return entry.inFlight;
  }

  const inFlight = fetcher().then((data) => {
    cache.set(key, { data, updatedAt: Date.now() });
    return data;
  });
  cache.set(key, { data: entry?.data as T, updatedAt: entry?.updatedAt ?? 0, inFlight });
  return inFlight;
}

export function resetSWRCache(): void {
  cache.clear();
}

export function getSWRCacheStats(key: string): { size: number; hasEntry: boolean } {
  return { size: cache.size, hasEntry: cache.has(key) };
}
