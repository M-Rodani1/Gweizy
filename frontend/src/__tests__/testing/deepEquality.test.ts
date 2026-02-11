import { describe, it, expect } from 'vitest';
import { deepEqual } from '../../utils/deepEqual';

describe('deepEqual', () => {
  it('compares primitive values correctly', () => {
    expect(deepEqual(1, 1)).toBe(true);
    expect(deepEqual('a', 'a')).toBe(true);
    expect(deepEqual('a', 'b')).toBe(false);
    expect(deepEqual(NaN, NaN)).toBe(true);
    expect(deepEqual(1, '1')).toBe(false);
  });

  it('compares arrays deeply', () => {
    expect(deepEqual([1, 2, 3], [1, 2, 3])).toBe(true);
    expect(deepEqual([1, 2, 3], [1, 2, 4])).toBe(false);
    expect(deepEqual([1, [2, 3]], [1, [2, 3]])).toBe(true);
    expect(deepEqual([1, 2], [1, 2, 3])).toBe(false);
  });

  it('compares objects deeply', () => {
    expect(deepEqual({ a: 1 }, { a: 1 })).toBe(true);
    expect(deepEqual({ a: 1 }, { a: 2 })).toBe(false);
    expect(deepEqual({ a: { b: 2 } }, { a: { b: 2 } })).toBe(true);
    expect(deepEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
  });

  it('compares Date values by timestamp', () => {
    const first = new Date('2024-01-01T00:00:00Z');
    const second = new Date('2024-01-01T00:00:00Z');
    const third = new Date('2024-01-02T00:00:00Z');

    expect(deepEqual(first, second)).toBe(true);
    expect(deepEqual(first, third)).toBe(false);
  });

  it('returns false for null and object mismatches', () => {
    expect(deepEqual(null, null)).toBe(true);
    expect(deepEqual(null, {})).toBe(false);
    expect(deepEqual({}, null)).toBe(false);
  });
});
