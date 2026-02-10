import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { getWithSWR, resetSWRCache } from '../../utils/swrCache';

describe('stale-while-revalidate cache', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2024-01-01T00:00:00Z'));
    resetSWRCache();
  });

  afterEach(() => {
    vi.useRealTimers();
    resetSWRCache();
  });

  it('returns cached data while within max age', async () => {
    const fetcher = vi.fn().mockResolvedValue('fresh');

    const first = await getWithSWR('key', fetcher, { maxAgeMs: 1000, staleWhileRevalidateMs: 1000 });
    expect(first).toBe('fresh');
    expect(fetcher).toHaveBeenCalledTimes(1);

    vi.setSystemTime(new Date('2024-01-01T00:00:00.500Z'));

    const second = await getWithSWR('key', fetcher, { maxAgeMs: 1000, staleWhileRevalidateMs: 1000 });
    expect(second).toBe('fresh');
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('returns stale data and revalidates in the background', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce('v1').mockResolvedValueOnce('v2');

    const initial = await getWithSWR('key', fetcher, { maxAgeMs: 1000, staleWhileRevalidateMs: 2000 });
    expect(initial).toBe('v1');

    vi.setSystemTime(new Date('2024-01-01T00:00:01.500Z'));

    const stale = await getWithSWR('key', fetcher, { maxAgeMs: 1000, staleWhileRevalidateMs: 2000 });
    expect(stale).toBe('v1');
    expect(fetcher).toHaveBeenCalledTimes(2);

    await Promise.resolve();

    const refreshed = await getWithSWR('key', fetcher, { maxAgeMs: 1000, staleWhileRevalidateMs: 2000 });
    expect(refreshed).toBe('v2');
  });

  it('waits for a fresh response when beyond stale window', async () => {
    const fetcher = vi.fn().mockResolvedValueOnce('v1').mockResolvedValueOnce('v2');

    await getWithSWR('key', fetcher, { maxAgeMs: 1000, staleWhileRevalidateMs: 1000 });

    vi.setSystemTime(new Date('2024-01-01T00:00:02.500Z'));

    const updated = await getWithSWR('key', fetcher, { maxAgeMs: 1000, staleWhileRevalidateMs: 1000 });
    expect(updated).toBe('v2');
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
