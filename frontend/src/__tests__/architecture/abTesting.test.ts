import { describe, it, expect, vi, beforeEach } from 'vitest';
import { assignVariant, getAssignedVariant } from '../../utils/abTesting';

describe('ab testing infrastructure', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn()
    });
  });

  it('assigns deterministic variants and persists', () => {
    const variant = assignVariant({ key: 'new-cta', trafficSplit: 50 }, 'user-1');
    expect(['A', 'B']).toContain(variant);
    expect(localStorage.setItem).toHaveBeenCalled();
  });

  it('returns stored assignment when available', () => {
    (localStorage.getItem as ReturnType<typeof vi.fn>).mockReturnValue('{"new-cta":"B"}');
    expect(getAssignedVariant('new-cta')).toBe('B');
  });
});
