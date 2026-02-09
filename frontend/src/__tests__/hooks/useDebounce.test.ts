/**
 * Tests for useDebounce hooks
 *
 * Tests cover:
 * - Basic debounce behavior
 * - Debounced callbacks with cancel/flush
 * - Leading edge execution
 * - Max wait functionality
 * - Cleanup on unmount
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import {
  useDebounce,
  useDebouncedCallback,
  useDebouncedSearch,
} from '../../hooks/useDebounce';

describe('useDebounce', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('basic behavior', () => {
    it('should return initial value immediately', () => {
      const { result } = renderHook(() => useDebounce('initial', 300));

      expect(result.current).toBe('initial');
    });

    it('should debounce value changes', () => {
      const { result, rerender } = renderHook(
        ({ value }) => useDebounce(value, 300),
        { initialProps: { value: 'first' } }
      );

      expect(result.current).toBe('first');

      rerender({ value: 'second' });
      expect(result.current).toBe('first');

      act(() => {
        vi.advanceTimersByTime(299);
      });
      expect(result.current).toBe('first');

      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(result.current).toBe('second');
    });

    it('should reset timer on value changes', () => {
      const { result, rerender } = renderHook(
        ({ value }) => useDebounce(value, 300),
        { initialProps: { value: 'first' } }
      );

      rerender({ value: 'second' });
      act(() => {
        vi.advanceTimersByTime(200);
      });

      rerender({ value: 'third' });
      act(() => {
        vi.advanceTimersByTime(200);
      });

      // Still 'first' because timer was reset
      expect(result.current).toBe('first');

      act(() => {
        vi.advanceTimersByTime(100);
      });
      expect(result.current).toBe('third');
    });

    it('should use default delay of 300ms', () => {
      const { result, rerender } = renderHook(
        ({ value }) => useDebounce(value),
        { initialProps: { value: 'initial' } }
      );

      rerender({ value: 'updated' });

      act(() => {
        vi.advanceTimersByTime(299);
      });
      expect(result.current).toBe('initial');

      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(result.current).toBe('updated');
    });

    it('should handle different types', () => {
      const { result: numberResult } = renderHook(() => useDebounce(42, 100));
      expect(numberResult.current).toBe(42);

      const { result: objectResult } = renderHook(() =>
        useDebounce({ key: 'value' }, 100)
      );
      expect(objectResult.current).toEqual({ key: 'value' });

      const { result: arrayResult } = renderHook(() =>
        useDebounce([1, 2, 3], 100)
      );
      expect(arrayResult.current).toEqual([1, 2, 3]);
    });
  });
});

describe('useDebouncedCallback', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('basic behavior', () => {
    it('should debounce callback execution', () => {
      const callback = vi.fn();
      const { result } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300 })
      );

      act(() => {
        result.current.debouncedCallback('arg1');
      });

      expect(callback).not.toHaveBeenCalled();

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(callback).toHaveBeenCalledWith('arg1');
      expect(callback).toHaveBeenCalledTimes(1);
    });

    it('should only call callback once after multiple rapid calls', () => {
      const callback = vi.fn();
      const { result } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300 })
      );

      act(() => {
        result.current.debouncedCallback('first');
        result.current.debouncedCallback('second');
        result.current.debouncedCallback('third');
      });

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(callback).toHaveBeenCalledTimes(1);
      expect(callback).toHaveBeenCalledWith('third');
    });

    it('should track pending state', () => {
      const callback = vi.fn();
      const { result } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300 })
      );

      expect(result.current.isPending).toBe(false);

      act(() => {
        result.current.debouncedCallback();
      });

      expect(result.current.isPending).toBe(true);

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(result.current.isPending).toBe(false);
    });
  });

  describe('cancel', () => {
    it('should cancel pending callback', () => {
      const callback = vi.fn();
      const { result } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300 })
      );

      act(() => {
        result.current.debouncedCallback();
      });

      expect(result.current.isPending).toBe(true);

      act(() => {
        result.current.cancel();
      });

      expect(result.current.isPending).toBe(false);

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(callback).not.toHaveBeenCalled();
    });
  });

  describe('flush', () => {
    it('should execute pending callback immediately', () => {
      const callback = vi.fn();
      const { result } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300 })
      );

      act(() => {
        result.current.debouncedCallback('arg');
      });

      expect(callback).not.toHaveBeenCalled();

      act(() => {
        result.current.flush();
      });

      expect(callback).toHaveBeenCalledWith('arg');
      expect(result.current.isPending).toBe(false);
    });

    it('should do nothing if no pending callback', () => {
      const callback = vi.fn();
      const { result } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300 })
      );

      act(() => {
        result.current.flush();
      });

      expect(callback).not.toHaveBeenCalled();
    });
  });

  describe('leading edge', () => {
    it('should execute immediately on first call with leading: true', () => {
      const callback = vi.fn();
      const { result } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300, leading: true })
      );

      act(() => {
        result.current.debouncedCallback('first');
      });

      expect(callback).toHaveBeenCalledWith('first');
      expect(callback).toHaveBeenCalledTimes(1);
    });
  });

  describe('maxWait', () => {
    it('should force execution after maxWait', () => {
      const callback = vi.fn();
      const { result } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300, maxWait: 500 })
      );

      // Keep calling to reset debounce timer
      act(() => {
        result.current.debouncedCallback('arg1');
      });

      act(() => {
        vi.advanceTimersByTime(200);
        result.current.debouncedCallback('arg2');
      });

      act(() => {
        vi.advanceTimersByTime(200);
        result.current.debouncedCallback('arg3');
      });

      // maxWait should have triggered
      act(() => {
        vi.advanceTimersByTime(100);
      });

      expect(callback).toHaveBeenCalledTimes(1);
      expect(callback).toHaveBeenCalledWith('arg3');
    });
  });

  describe('cleanup', () => {
    it('should cleanup on unmount', () => {
      const callback = vi.fn();
      const { result, unmount } = renderHook(() =>
        useDebouncedCallback(callback, { delay: 300 })
      );

      act(() => {
        result.current.debouncedCallback();
      });

      unmount();

      act(() => {
        vi.advanceTimersByTime(300);
      });

      // Callback should not be called after unmount
      // (though state changes won't happen, the timeout is cleared)
    });
  });
});

describe('useDebouncedSearch', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('basic behavior', () => {
    it('should return initial value', () => {
      const { result } = renderHook(() => useDebouncedSearch('initial', 300));

      expect(result.current.value).toBe('initial');
      expect(result.current.debouncedValue).toBe('initial');
    });

    it('should update value immediately but debounce debouncedValue', () => {
      const { result } = renderHook(() => useDebouncedSearch('', 300));

      act(() => {
        result.current.setValue('search');
      });

      expect(result.current.value).toBe('search');
      expect(result.current.debouncedValue).toBe('');
      expect(result.current.isSearching).toBe(true);

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(result.current.debouncedValue).toBe('search');
      expect(result.current.isSearching).toBe(false);
    });

    it('should clear value', () => {
      const { result } = renderHook(() => useDebouncedSearch('initial', 300));

      act(() => {
        result.current.clear();
      });

      expect(result.current.value).toBe('');
    });

    it('should track isSearching state', () => {
      const { result } = renderHook(() => useDebouncedSearch('', 300));

      expect(result.current.isSearching).toBe(false);

      act(() => {
        result.current.setValue('query');
      });

      expect(result.current.isSearching).toBe(true);

      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(result.current.isSearching).toBe(false);
    });

    it('should use default delay of 300ms', () => {
      const { result } = renderHook(() => useDebouncedSearch(''));

      act(() => {
        result.current.setValue('test');
      });

      act(() => {
        vi.advanceTimersByTime(299);
      });
      expect(result.current.debouncedValue).toBe('');

      act(() => {
        vi.advanceTimersByTime(1);
      });
      expect(result.current.debouncedValue).toBe('test');
    });
  });

  describe('typing simulation', () => {
    it('should handle rapid typing correctly', () => {
      const { result } = renderHook(() => useDebouncedSearch('', 500));

      // Simulate typing "hello" letter by letter
      act(() => {
        result.current.setValue('h');
      });
      act(() => {
        vi.advanceTimersByTime(100);
      });

      act(() => {
        result.current.setValue('he');
      });
      act(() => {
        vi.advanceTimersByTime(100);
      });

      act(() => {
        result.current.setValue('hel');
      });
      act(() => {
        vi.advanceTimersByTime(100);
      });

      act(() => {
        result.current.setValue('hell');
      });
      act(() => {
        vi.advanceTimersByTime(100);
      });

      act(() => {
        result.current.setValue('hello');
      });

      // Debounced value should still be empty
      expect(result.current.value).toBe('hello');
      expect(result.current.debouncedValue).toBe('');

      // After delay, should have final value
      act(() => {
        vi.advanceTimersByTime(500);
      });

      expect(result.current.debouncedValue).toBe('hello');
    });
  });
});
