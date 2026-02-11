import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import BestTimeWidget from '../../components/BestTimeWidget';

describe('useMemo/useCallback audit', () => {
  it('renders derived savings using memoized hourly stats', () => {
    render(<BestTimeWidget currentGas={0.002} />);

    expect(screen.getByText('Best Times to Transact')).toBeInTheDocument();
    expect(screen.getByText(/Save up to 57% vs peak hours/)).toBeInTheDocument();
  });
});
