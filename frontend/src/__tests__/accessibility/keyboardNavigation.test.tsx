import { describe, it, expect } from 'vitest';
import { render, fireEvent, screen } from '@testing-library/react';
import { useChartKeyboardNav } from '../../hooks/useChartKeyboardNav';

function ChartNavigationTest() {
  const dataPoints = [
    { label: 'A', value: '1' },
    { label: 'B', value: '2' },
    { label: 'C', value: '3' },
  ];
  const { containerProps, focusedIndex } = useChartKeyboardNav({
    dataPoints,
    chartLabel: 'Test chart',
  });

  return (
    <div data-testid="chart" {...containerProps}>
      <span data-testid="focused">{focusedIndex}</span>
    </div>
  );
}

describe('keyboard navigation', () => {
  it('moves focus with arrow keys', () => {
    render(<ChartNavigationTest />);
    const chart = screen.getByTestId('chart');

    fireEvent.focus(chart);
    expect(screen.getByTestId('focused').textContent).toBe('0');

    fireEvent.keyDown(chart, { key: 'ArrowRight' });
    expect(screen.getByTestId('focused').textContent).toBe('1');

    fireEvent.keyDown(chart, { key: 'ArrowLeft' });
    expect(screen.getByTestId('focused').textContent).toBe('0');
  });

  it('moves to first and last with Home/End', () => {
    render(<ChartNavigationTest />);
    const chart = screen.getByTestId('chart');

    fireEvent.focus(chart);
    fireEvent.keyDown(chart, { key: 'End' });
    expect(screen.getByTestId('focused').textContent).toBe('2');

    fireEvent.keyDown(chart, { key: 'Home' });
    expect(screen.getByTestId('focused').textContent).toBe('0');
  });
});
