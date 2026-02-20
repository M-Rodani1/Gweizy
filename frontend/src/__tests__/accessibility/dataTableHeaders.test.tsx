import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import GasPriceTable from '../../components/GasPriceTable';
import UserTransactionHistory from '../../components/UserTransactionHistory';
import { fetchTransactions, fetchUserHistory } from '../../api/gasApi';

vi.mock('../../api/gasApi', () => ({
  fetchTransactions: vi.fn(),
  fetchUserHistory: vi.fn(),
}));

const mockedFetchTransactions = vi.mocked(fetchTransactions);
const mockedFetchUserHistory = vi.mocked(fetchUserHistory);

describe('data table accessibility', () => {
  it('adds column and row headers to the gas price table', async () => {
    mockedFetchTransactions.mockResolvedValueOnce([
      {
        txHash: '0xabc',
        method: 'Swap',
        age: '1 min ago',
        gasUsed: 21000,
        gasPrice: 1.2345,
        timestamp: 1710000000,
      },
    ]);

    render(<GasPriceTable />);

    await screen.findByRole('table');
    expect(screen.getAllByRole('columnheader')).toHaveLength(5);
    expect(screen.getAllByRole('rowheader').length).toBeGreaterThan(0);
  });

  it('exposes row and column metadata for the virtualized table', async () => {
    const historyResponse = {
      address: '0x123',
      transactions: [
        {
          hash: '0x123',
          timestamp: 1710000000,
          gas_used: 21000,
          gas_price: 1000000000,
          gasUsed: 21000,
          gasPrice: 1000000000,
          value: '0',
          from: '0xfrom',
          to: '0xto',
        },
      ],
      total_savings: 0.01,
      total_transactions: 1,
      total_gas_paid: 0.01,
      potential_savings: 0.005,
      savings_percentage: 50,
      recommendations: {},
    };

    mockedFetchUserHistory.mockResolvedValueOnce(
      historyResponse as unknown as Awaited<ReturnType<typeof fetchUserHistory>>
    );

    render(<UserTransactionHistory address="0x123" />);

    const table = await screen.findByRole('table', { name: /recent transactions/i });
    expect(table).toHaveAttribute('aria-rowcount', '2');
    expect(table).toHaveAttribute('aria-colcount', '5');
    expect(screen.getAllByRole('rowheader').length).toBeGreaterThan(0);
  });
});
