import { describe, it, expect } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { createRef } from 'react';
import ErrorBoundary from '../../components/ErrorBoundary';

describe('ErrorBoundary reset keys', () => {
  it('resets when resetKeys change', () => {
    const ref = createRef<ErrorBoundary>();
    const { rerender } = render(
      <ErrorBoundary resetKeys={[1]} ref={ref}>
        <div>Safe</div>
      </ErrorBoundary>
    );

    act(() => {
      ref.current?.setState({ hasError: true, error: new Error('Boom'), errorInfo: null });
    });

    expect(screen.getByRole('alert')).toBeInTheDocument();

    rerender(
      <ErrorBoundary resetKeys={[2]} ref={ref}>
        <div>Safe</div>
      </ErrorBoundary>
    );

    expect(screen.getByText('Safe')).toBeInTheDocument();
  });
});
