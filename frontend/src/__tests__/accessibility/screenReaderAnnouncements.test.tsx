import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useChartKeyboardNav } from '../../hooks/useChartKeyboardNav';

describe('screen reader announcements', () => {
  it('announces the focused chart point', () => {
    const dataPoints = [
      { label: 'Point A', value: '10 gwei' },
      { label: 'Point B', value: '12 gwei' },
    ];

    const { result } = renderHook(() =>
      useChartKeyboardNav({ dataPoints, chartLabel: 'Gas chart' })
    );

    act(() => {
      result.current.handleFocus();
    });

    expect(result.current.announcement).toContain('Point A: 10 gwei');
    expect(result.current.announcement).toContain('Point 1 of 2');
  });
});
