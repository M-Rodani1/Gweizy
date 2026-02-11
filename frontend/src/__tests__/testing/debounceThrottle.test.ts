import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { debounce, throttle } from '../../utils/debounce';

describe('debounce utility', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('delays execution until the debounce window elapses', () => {
    const spy = vi.fn();
    const debounced = debounce(spy, 200);

    debounced('first');
    debounced('second');

    expect(spy).not.toHaveBeenCalled();

    vi.advanceTimersByTime(199);
    expect(spy).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith('second');
  });
});

describe('throttle utility', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(0));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('limits calls to the throttle window', () => {
    const spy = vi.fn();
    const throttled = throttle(spy, 1000);

    throttled('first');
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith('first');

    vi.setSystemTime(new Date(500));
    throttled('second');
    expect(spy).toHaveBeenCalledTimes(1);

    vi.setSystemTime(new Date(1000));
    throttled('third');
    expect(spy).toHaveBeenCalledTimes(2);
    expect(spy).toHaveBeenLastCalledWith('third');
  });
});
