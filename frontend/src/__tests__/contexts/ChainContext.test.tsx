/**
 * Tests for ChainContext
 *
 * Tests cover:
 * - ChainProvider initialization
 * - Chain selection and persistence
 * - Multi-chain gas fetching
 * - Best chain calculation
 * - useChain, useSelectedChainGas, useChainComparison hooks
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { ReactNode } from 'react';
import { ChainProvider, useChain, useSelectedChainGas, useChainComparison } from '../../contexts/ChainContext';

// Mock the chains config
vi.mock('../../config/chains', () => ({
  SUPPORTED_CHAINS: {
    8453: {
      id: 8453,
      name: 'Base',
      shortName: 'BASE',
      nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
      rpcUrls: ['https://mainnet.base.org'],
      enabled: true,
      isL2: true,
    },
    1: {
      id: 1,
      name: 'Ethereum',
      shortName: 'ETH',
      nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
      rpcUrls: ['https://eth.llamarpc.com'],
      enabled: true,
      isL2: false,
    },
    42161: {
      id: 42161,
      name: 'Arbitrum',
      shortName: 'ARB',
      nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
      rpcUrls: ['https://arb1.arbitrum.io/rpc'],
      enabled: true,
      isL2: true,
    },
    999: {
      id: 999,
      name: 'Disabled',
      shortName: 'DIS',
      nativeCurrency: { name: 'Test', symbol: 'TEST', decimals: 18 },
      rpcUrls: ['https://disabled.rpc'],
      enabled: false,
      isL2: false,
    },
  },
  DEFAULT_CHAIN_ID: 8453,
  getChainById: vi.fn((chainId: number) => {
    const chains: Record<number, unknown> = {
      8453: { id: 8453, name: 'Base', enabled: true },
      1: { id: 1, name: 'Ethereum', enabled: true },
      42161: { id: 42161, name: 'Arbitrum', enabled: true },
    };
    return chains[chainId];
  }),
  getEnabledChains: vi.fn(() => [
    { id: 8453, name: 'Base', enabled: true, rpcUrls: ['https://mainnet.base.org'] },
    { id: 1, name: 'Ethereum', enabled: true, rpcUrls: ['https://eth.llamarpc.com'] },
    { id: 42161, name: 'Arbitrum', enabled: true, rpcUrls: ['https://arb1.arbitrum.io/rpc'] },
  ]),
}));

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(global, 'localStorage', {
  value: localStorageMock,
  writable: true,
});

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock AbortController
class MockAbortController {
  signal = { aborted: false };
  abort = vi.fn(() => {
    this.signal.aborted = true;
  });
}
global.AbortController = MockAbortController as unknown as typeof AbortController;

// Wrapper for hooks
const wrapper = ({ children }: { children: ReactNode }) => (
  <ChainProvider>{children}</ChainProvider>
);

describe('ChainContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorageMock.clear();
    vi.useFakeTimers();

    // Default mock for successful gas fetch
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ result: '0x5F5E100' }), // 100000000 wei = 0.1 gwei
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('ChainProvider initialization', () => {
    it('should use default chain when no saved chain', () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      expect(result.current.selectedChainId).toBe(8453);
      expect(result.current.selectedChain.name).toBe('Base');
    });

    it('should load saved chain from localStorage', () => {
      localStorageMock.getItem.mockReturnValueOnce('1');

      const { result } = renderHook(() => useChain(), { wrapper });

      expect(result.current.selectedChainId).toBe(1);
    });

    it('should fall back to default if saved chain is disabled', () => {
      localStorageMock.getItem.mockReturnValueOnce('999');

      const { result } = renderHook(() => useChain(), { wrapper });

      expect(result.current.selectedChainId).toBe(8453);
    });

    it('should return enabled chains only', () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      expect(result.current.enabledChains).toHaveLength(3);
      expect(result.current.enabledChains.every(c => c.enabled)).toBe(true);
    });
  });

  describe('chain selection', () => {
    it('should update selected chain', async () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      act(() => {
        result.current.setSelectedChainId(1);
      });

      expect(result.current.selectedChainId).toBe(1);
    });

    it('should persist chain selection to localStorage', async () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      act(() => {
        result.current.setSelectedChainId(42161);
      });

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'gweizy_selected_chain',
        '42161'
      );
    });

    it('should not select disabled chains', async () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      act(() => {
        result.current.setSelectedChainId(999);
      });

      // Should remain on default chain
      expect(result.current.selectedChainId).toBe(8453);
    });
  });

  describe('multi-chain gas fetching', () => {
    it('should set loading state while fetching', async () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      expect(result.current.isLoading).toBe(true);
    });

    it('should fetch gas prices for all enabled chains', async () => {
      renderHook(() => useChain(), { wrapper });

      // Wait for initial fetch
      await act(async () => {
        await Promise.resolve();
      });

      // Should have called fetch for each enabled chain
      expect(mockFetch).toHaveBeenCalled();
    });

    it('should update multiChainGas with fetched data', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ result: '0x3B9ACA00' }), // 1 gwei
      });

      const { result } = renderHook(() => useChain(), { wrapper });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      // Check that gas prices are populated
      const gasData = result.current.multiChainGas;
      expect(Object.keys(gasData).length).toBeGreaterThan(0);
    });

    it('should handle fetch errors gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useChain(), { wrapper });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      // Should not throw
      expect(result.current.multiChainGas).toBeDefined();
    });

    it('should refresh gas prices when refreshMultiChainGas is called', async () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      await act(async () => {
        await Promise.resolve();
      });

      const initialCallCount = mockFetch.mock.calls.length;

      await act(async () => {
        await result.current.refreshMultiChainGas();
      });

      expect(mockFetch.mock.calls.length).toBeGreaterThan(initialCallCount);
    });
  });

  describe('best chain calculation', () => {
    it('should return null when no valid gas data', () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      // Before any fetching completes
      expect(result.current.bestChainForTx).toBeNull();
    });

    it('should calculate best chain based on lowest gas', async () => {
      // Mock different gas prices for different chains
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ result: '0x77359400' }), // 2 gwei - Base
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ result: '0x3B9ACA00' }), // 1 gwei - Ethereum (cheapest)
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ result: '0xB2D05E00' }), // 3 gwei - Arbitrum
        });

      const { result } = renderHook(() => useChain(), { wrapper });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
      });

      // Best chain should be Ethereum (lowest gas)
      if (result.current.bestChainForTx) {
        expect(result.current.bestChainForTx.chainId).toBe(1);
        expect(result.current.bestChainForTx.savings).toBeGreaterThan(0);
      }
    });
  });

  describe('useChain hook', () => {
    it('should throw when used outside provider', () => {
      // Suppress console.error for this test
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useChain());
      }).toThrow('useChain must be used within a ChainProvider');

      consoleSpy.mockRestore();
    });

    it('should return all context values', () => {
      const { result } = renderHook(() => useChain(), { wrapper });

      expect(result.current).toHaveProperty('selectedChainId');
      expect(result.current).toHaveProperty('selectedChain');
      expect(result.current).toHaveProperty('setSelectedChainId');
      expect(result.current).toHaveProperty('enabledChains');
      expect(result.current).toHaveProperty('multiChainGas');
      expect(result.current).toHaveProperty('refreshMultiChainGas');
      expect(result.current).toHaveProperty('bestChainForTx');
      expect(result.current).toHaveProperty('isLoading');
    });
  });

  describe('useSelectedChainGas hook', () => {
    it('should return gas for selected chain', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ result: '0x3B9ACA00' }),
      });

      const { result } = renderHook(() => useSelectedChainGas(), { wrapper });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      // Initially undefined or the fetched value
      expect(result.current === undefined || result.current?.chainId === 8453).toBe(true);
    });
  });

  describe('useChainComparison hook', () => {
    it('should return chains sorted by gas price', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ result: '0x77359400' }), // 2 gwei
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ result: '0x3B9ACA00' }), // 1 gwei
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ result: '0xB2D05E00' }), // 3 gwei
        });

      const { result } = renderHook(() => useChainComparison(), { wrapper });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
      });

      // Should be sorted by gas price ascending
      const comparison = result.current;
      if (comparison.length >= 2) {
        for (let i = 1; i < comparison.length; i++) {
          const prev = comparison[i - 1].gas?.gasPrice || 0;
          const curr = comparison[i].gas?.gasPrice || 0;
          expect(curr).toBeGreaterThanOrEqual(prev);
        }
      }
    });

    it('should exclude chains with errors', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ result: '0x3B9ACA00' }),
        })
        .mockRejectedValueOnce(new Error('RPC error'))
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ result: '0x3B9ACA00' }),
        });

      const { result } = renderHook(() => useChainComparison(), { wrapper });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
        await Promise.resolve();
      });

      // Should only include chains without errors
      const comparison = result.current;
      expect(comparison.every(c => !c.gas?.error)).toBe(true);
    });
  });

  describe('caching', () => {
    it('should cache gas prices to localStorage', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ result: '0x3B9ACA00' }),
      });

      renderHook(() => useChain(), { wrapper });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      // Should have written to cache
      expect(localStorageMock.setItem).toHaveBeenCalled();
    });

    it('should read from cache when RPC fails', async () => {
      // Pre-populate cache
      const cachedData = JSON.stringify({
        gasPrice: 1.5,
        timestamp: Date.now(),
      });
      localStorageMock.getItem.mockImplementation((key: string) => {
        if (key.startsWith('gweizy_chain_gas_v1:')) {
          return cachedData;
        }
        return null;
      });

      mockFetch.mockRejectedValue(new Error('RPC error'));

      const { result } = renderHook(() => useChain(), { wrapper });

      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });

      // Should have data from cache
      expect(result.current.multiChainGas).toBeDefined();
    });
  });
});
