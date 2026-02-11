import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import HourlyHeatmap from '../../components/HourlyHeatmap';

describe('CSS containment', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false }));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('applies layout containment to heavy widgets', async () => {
    let container: HTMLElement | null = null;

    await act(async () => {
      const result = render(<HourlyHeatmap />);
      container = result.container;
    });

    const root = container?.firstElementChild as HTMLElement | null;

    expect(root).toBeTruthy();
    expect(root?.className).toContain('contain-layout');
  });
});
