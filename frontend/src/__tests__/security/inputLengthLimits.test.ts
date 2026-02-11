import { describe, it, expect } from 'vitest';
import { enforceMaxLength, isWithinLength } from '../../utils/inputLimits';

describe('Input length limits', () => {
  it('truncates input beyond max length', () => {
    const value = 'a'.repeat(20);
    expect(enforceMaxLength(value, 10)).toBe('a'.repeat(10));
  });

  it('keeps input within bounds', () => {
    expect(enforceMaxLength('safe', 10)).toBe('safe');
    expect(isWithinLength('safe', 10)).toBe(true);
  });

  it('rejects invalid max length values', () => {
    expect(enforceMaxLength('value', 0)).toBe('');
    expect(isWithinLength('value', 0)).toBe(false);
  });
});
