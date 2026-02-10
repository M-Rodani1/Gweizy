import { describe, it, expect, vi } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import AppShell from '../../components/layout/AppShell';

vi.mock('../../contexts/ChainContext', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../contexts/ChainContext')>();
  const chain = {
    id: 1,
    name: 'Ethereum',
    shortName: 'ETH',
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: [],
    blockExplorer: '',
    icon: 'E',
    color: '#627EEA',
    isL2: false,
    enabled: true,
    gasUnit: 'gwei',
    avgBlockTime: 12,
    supportsEIP1559: true
  };
  const chainGas = {
    chainId: chain.id,
    gasPrice: 12,
    timestamp: 1700000000000,
    loading: false,
    error: null
  };
  const value = {
    selectedChainId: chain.id,
    selectedChain: chain,
    setSelectedChainId: vi.fn(),
    enabledChains: [chain],
    multiChainGas: { [chain.id]: chainGas },
    refreshMultiChainGas: vi.fn().mockResolvedValue(undefined),
    bestChainForTx: null,
    isLoading: false
  };

  return {
    ...actual,
    useChain: () => value,
    useChainComparison: () => [{ chain, gas: chainGas }]
  };
});

vi.mock('../../api/gasApi', () => ({
  checkHealth: vi.fn().mockResolvedValue(true)
}));

describe('landmark regions', () => {
  it('renders banner, navigation, complementary, and main landmarks', async () => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn()
    });
    vi.stubGlobal('matchMedia', vi.fn().mockImplementation(() => ({
      matches: false,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    })));

    await act(async () => {
      render(
        <MemoryRouter initialEntries={['/app']}>
          <AppShell>
            <div>Content</div>
          </AppShell>
        </MemoryRouter>
      );
    });

    expect(screen.getByRole('banner')).toBeInTheDocument();
    expect(screen.getByRole('navigation')).toBeInTheDocument();
    expect(screen.getByRole('complementary', { name: /application navigation/i })).toBeInTheDocument();
    expect(screen.getByRole('main')).toHaveAttribute('id', 'main-content');
  });
});
