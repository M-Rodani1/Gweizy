import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { rateLimiter } from '../../utils/rateLimiter';

describe('rateLimiter', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2024-01-01T00:00:00Z'));
    rateLimiter.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
    rateLimiter.clear();
  });

  it('allows requests within the limit', () => {
    expect(rateLimiter.isAllowed('endpoint', 2, 1000)).toBe(true);
    expect(rateLimiter.isAllowed('endpoint', 2, 1000)).toBe(true);
    expect(rateLimiter.isAllowed('endpoint', 2, 1000)).toBe(false);
  });

  it('resets after the window elapses', () => {
    rateLimiter.isAllowed('endpoint', 1, 1000);
    expect(rateLimiter.isAllowed('endpoint', 1, 1000)).toBe(false);

    vi.setSystemTime(new Date('2024-01-01T00:00:01.500Z'));
    expect(rateLimiter.isAllowed('endpoint', 1, 1000)).toBe(true);
  });

  it('returns time until reset', () => {
    rateLimiter.isAllowed('endpoint', 1, 2000);
    vi.setSystemTime(new Date('2024-01-01T00:00:00.500Z'));
    const remaining = rateLimiter.getTimeUntilReset('endpoint');
    expect(remaining).toBeGreaterThan(0);
    expect(remaining).toBeLessThanOrEqual(1500);
  });
});
