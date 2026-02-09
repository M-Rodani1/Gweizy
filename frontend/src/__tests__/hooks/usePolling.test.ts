/**
 * Tests for usePolling hook
 *
 * Tests cover:
 * - Initial fetch behavior
 * - Polling interval timing
 * - Pause/resume functionality
 * - Error handling
 * - Cleanup on unmount
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { usePolling } from '../../hooks/usePolling';

describe('usePolling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  describe('initial behavior', () => {
    it('should fetch immediately by default', async () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      renderHook(() => usePolling({ fetcher }));

      expect(fetcher).toHaveBeenCalledTimes(1);
    });

    it('should not fetch immediately when immediate is false', () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      renderHook(() => usePolling({ fetcher, immediate: false }));

      expect(fetcher).not.toHaveBeenCalled();
    });

    it('should start with loading true when immediate is true', () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      const { result } = renderHook(() => usePolling({ fetcher }));

      expect(result.current.loading).toBe(true);
    });

    it('should start with loading false when immediate is false', () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      const { result } = renderHook(() =>
        usePolling({ fetcher, immediate: false })
      );

      expect(result.current.loading).toBe(false);
    });

    it('should not fetch when enabled is false', () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      renderHook(() => usePolling({ fetcher, enabled: false }));

      expect(fetcher).not.toHaveBeenCalled();
    });
  });

  describe('data fetching', () => {
    it('should set data after successful fetch', async () => {
      const mockData = { value: 42 };
      const fetcher = vi.fn().mockResolvedValue(mockData);

      const { result } = renderHook(() => usePolling({ fetcher }));

      // Flush promises
      await act(async () => {
        await Promise.resolve();
      });

      expect(result.current.data).toEqual(mockData);
      expect(result.current.loading).toBe(false);
      expect(result.current.error).toBeNull();
    });

    it('should call onSuccess callback with data', async () => {
      const mockData = { value: 42 };
      const fetcher = vi.fn().mockResolvedValue(mockData);
      const onSuccess = vi.fn();

      renderHook(() => usePolling({ fetcher, onSuccess }));

      await act(async () => {
        await Promise.resolve();
      });

      expect(onSuccess).toHaveBeenCalledWith(mockData);
    });

    it('should set error on fetch failure', async () => {
      const error = new Error('Fetch failed');
      const fetcher = vi.fn().mockRejectedValue(error);

      const { result } = renderHook(() => usePolling({ fetcher }));

      await act(async () => {
        await Promise.resolve();
      });

      expect(result.current.error).toEqual(error);
      expect(result.current.loading).toBe(false);
      expect(result.current.data).toBeNull();
    });

    it('should call onError callback on failure', async () => {
      const error = new Error('Fetch failed');
      const fetcher = vi.fn().mockRejectedValue(error);
      const onError = vi.fn();

      renderHook(() => usePolling({ fetcher, onError }));

      await act(async () => {
        await Promise.resolve();
      });

      expect(onError).toHaveBeenCalledWith(error);
    });

    it('should convert non-Error to Error in onError', async () => {
      const fetcher = vi.fn().mockRejectedValue('string error');
      const onError = vi.fn();

      renderHook(() => usePolling({ fetcher, onError }));

      await act(async () => {
        await Promise.resolve();
      });

      expect(onError).toHaveBeenCalledWith(expect.any(Error));
      expect(onError.mock.calls[0][0].message).toBe('string error');
    });
  });

  describe('polling interval', () => {
    it('should poll at the specified interval', async () => {
      const fetcher = vi.fn().mockResolvedValue('data');
      const interval = 5000;

      renderHook(() => usePolling({ fetcher, interval }));

      // Initial fetch
      expect(fetcher).toHaveBeenCalledTimes(1);

      // Advance to first interval
      await act(async () => {
        vi.advanceTimersByTime(interval);
      });

      expect(fetcher).toHaveBeenCalledTimes(2);

      // Advance to second interval
      await act(async () => {
        vi.advanceTimersByTime(interval);
      });

      expect(fetcher).toHaveBeenCalledTimes(3);
    });

    it('should use default interval of 30000ms', async () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      renderHook(() => usePolling({ fetcher }));

      expect(fetcher).toHaveBeenCalledTimes(1);

      // Advance less than default interval
      await act(async () => {
        vi.advanceTimersByTime(29999);
      });

      expect(fetcher).toHaveBeenCalledTimes(1);

      // Advance to complete the interval
      await act(async () => {
        vi.advanceTimersByTime(1);
      });

      expect(fetcher).toHaveBeenCalledTimes(2);
    });
  });

  describe('pause and resume', () => {
    it('should stop polling when paused', async () => {
      const fetcher = vi.fn().mockResolvedValue('data');
      const interval = 5000;

      const { result } = renderHook(() => usePolling({ fetcher, interval }));

      expect(fetcher).toHaveBeenCalledTimes(1);

      // Pause polling
      act(() => {
        result.current.pause();
      });

      expect(result.current.isPolling).toBe(false);

      // Advance time - should not trigger fetch
      await act(async () => {
        vi.advanceTimersByTime(interval * 3);
      });

      expect(fetcher).toHaveBeenCalledTimes(1);
    });

    it('should resume polling when resumed', async () => {
      const fetcher = vi.fn().mockResolvedValue('data');
      const interval = 5000;

      const { result } = renderHook(() => usePolling({ fetcher, interval }));

      // Pause
      act(() => {
        result.current.pause();
      });

      // Resume
      await act(async () => {
        result.current.resume();
      });

      expect(result.current.isPolling).toBe(true);
      // Resume triggers an immediate fetch
      expect(fetcher).toHaveBeenCalledTimes(2);

      // Advance interval - should trigger another fetch
      await act(async () => {
        vi.advanceTimersByTime(interval);
      });

      expect(fetcher).toHaveBeenCalledTimes(3);
    });
  });

  describe('refresh', () => {
    it('should manually trigger a fetch', async () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      const { result } = renderHook(() =>
        usePolling({ fetcher, immediate: false })
      );

      expect(fetcher).not.toHaveBeenCalled();

      await act(async () => {
        await result.current.refresh();
      });

      expect(fetcher).toHaveBeenCalledTimes(1);
    });

    it('should update data on refresh', async () => {
      const fetcher = vi
        .fn()
        .mockResolvedValueOnce('first')
        .mockResolvedValueOnce('second');

      const { result } = renderHook(() => usePolling({ fetcher }));

      await act(async () => {
        await Promise.resolve();
      });

      expect(result.current.data).toBe('first');

      await act(async () => {
        await result.current.refresh();
      });

      expect(result.current.data).toBe('second');
    });
  });

  describe('cleanup', () => {
    it('should stop polling on unmount', async () => {
      const fetcher = vi.fn().mockResolvedValue('data');
      const interval = 5000;

      const { unmount } = renderHook(() => usePolling({ fetcher, interval }));

      expect(fetcher).toHaveBeenCalledTimes(1);

      unmount();

      // Advance time after unmount
      await act(async () => {
        vi.advanceTimersByTime(interval * 3);
      });

      // Should not have called after unmount
      expect(fetcher).toHaveBeenCalledTimes(1);
    });
  });

  describe('return values', () => {
    it('should return all expected properties', () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      const { result } = renderHook(() => usePolling({ fetcher }));

      expect(result.current).toHaveProperty('data');
      expect(result.current).toHaveProperty('loading');
      expect(result.current).toHaveProperty('error');
      expect(result.current).toHaveProperty('refresh');
      expect(result.current).toHaveProperty('pause');
      expect(result.current).toHaveProperty('resume');
      expect(result.current).toHaveProperty('isPolling');
    });

    it('should have correct initial isPolling state', () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      const { result } = renderHook(() => usePolling({ fetcher }));

      expect(result.current.isPolling).toBe(true);
    });

    it('should have isPolling false when enabled is false', () => {
      const fetcher = vi.fn().mockResolvedValue('data');

      const { result } = renderHook(() =>
        usePolling({ fetcher, enabled: false })
      );

      expect(result.current.isPolling).toBe(false);
    });
  });
});
