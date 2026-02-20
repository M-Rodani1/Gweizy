import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import GasPriceGraph from '../../components/GasPriceGraph';
import { fetchPredictions, fetchCurrentGas } from '../../api/gasApi';

vi.mock('../../api/gasApi', () => ({
  fetchPredictions: vi.fn(() => new Promise(() => {})),
  fetchCurrentGas: vi.fn(() => new Promise(() => {})),
}));

describe('progress indicator announcements', () => {
  it('announces gas price loading state', () => {
    render(<GasPriceGraph />);

    const status = screen.getByRole('status');
    expect(status).toHaveTextContent('Loading gas price data...');
    expect(fetchPredictions).toHaveBeenCalled();
    expect(fetchCurrentGas).toHaveBeenCalled();
  });
});
