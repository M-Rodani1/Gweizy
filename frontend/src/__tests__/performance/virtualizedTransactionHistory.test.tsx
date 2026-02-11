import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import UserTransactionHistory from '../../components/UserTransactionHistory';
import { fetchUserHistory } from '../../api/gasApi';

vi.mock('../../api/gasApi', () => ({
  fetchUserHistory: vi.fn(),
}));

const makeHash = (index: number) => `0x${index.toString(16).padStart(8, '0')}${'a'.repeat(56)}`;

const buildTransactions = (count: number) =>
  Array.from({ length: count }, (_, index) => ({
    hash: makeHash(index),
    timestamp: 1700000000 + index,
    gasUsed: 21000 + index,
    gasPrice: 1_000_000_000,
    value: '0',
    from: '0xfrom',
    to: '0xto',
  }));

describe('Virtualized transaction history table', () => {
  beforeEach(() => {
    vi.mocked(fetchUserHistory).mockResolvedValue({
      transactions: buildTransactions(50),
      total_transactions: 50,
      total_gas_paid: 1.23,
      potential_savings: 0.5,
      savings_percentage: 12,
      recommendations: {},
    });
  });

  it('renders a virtualized subset of rows for large histories', async () => {
    render(<UserTransactionHistory address="0x123" />);

    await screen.findByText('Your Recent Base Transactions');

    const transactions = buildTransactions(50);
    const firstLabel = `${transactions[0].hash.slice(0, 10)}...${transactions[0].hash.slice(-8)}`;
    const lastLabel = `${transactions[transactions.length - 1].hash.slice(0, 10)}...${transactions[transactions.length - 1].hash.slice(-8)}`;

    expect(screen.getByText(firstLabel)).toBeInTheDocument();
    expect(screen.queryByText(lastLabel)).toBeNull();

    const dataRows = screen.getAllByRole('row').filter((row) => row.querySelector('a'));
    expect(dataRows.length).toBeLessThan(transactions.length);
  });
});
