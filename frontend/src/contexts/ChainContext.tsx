import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { ChainConfig, SUPPORTED_CHAINS, DEFAULT_CHAIN_ID, getChainById, getEnabledChains } from '../config/chains';

interface MultiChainGas {
  chainId: number;
  gasPrice: number;
  timestamp: number;
  loading: boolean;
  error: string | null;
}

interface ChainContextType {
  // Current selected chain
  selectedChainId: number;
  selectedChain: ChainConfig;
  setSelectedChainId: (chainId: number) => void;

  // All enabled chains
  enabledChains: ChainConfig[];

  // Multi-chain gas prices (for comparison view)
  multiChainGas: Record<number, MultiChainGas>;
  refreshMultiChainGas: () => Promise<void>;

  // Best chain recommendation
  bestChainForTx: {
    chainId: number;
    savings: number;
    reason: string;
  } | null;

  // Loading states
  isLoading: boolean;
}

const ChainContext = createContext<ChainContextType | undefined>(undefined);

const STORAGE_KEY = 'gweizy_selected_chain';

export const ChainProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Load saved chain from localStorage or use default
  const [selectedChainId, setSelectedChainIdState] = useState<number>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const chainId = parseInt(saved, 10);
        if (SUPPORTED_CHAINS[chainId]?.enabled) {
          return chainId;
        }
      }
    }
    return DEFAULT_CHAIN_ID;
  });

  const [multiChainGas, setMultiChainGas] = useState<Record<number, MultiChainGas>>({});
  const [isLoading, setIsLoading] = useState(true);

  const enabledChains = getEnabledChains();
  const selectedChain = getChainById(selectedChainId) || SUPPORTED_CHAINS[DEFAULT_CHAIN_ID];

  // Persist selected chain
  const setSelectedChainId = useCallback((chainId: number) => {
    if (SUPPORTED_CHAINS[chainId]?.enabled) {
      setSelectedChainIdState(chainId);
      localStorage.setItem(STORAGE_KEY, chainId.toString());
    }
  }, []);

  // Fetch gas price for a single chain
  const fetchChainGas = useCallback(async (chainId: number): Promise<MultiChainGas> => {
    const chain = SUPPORTED_CHAINS[chainId];
    if (!chain) {
      return { chainId, gasPrice: 0, timestamp: Date.now(), loading: false, error: 'Unknown chain' };
    }

    try {
      // Try each RPC until one works
      for (const rpcUrl of chain.rpcUrls) {
        try {
          const response = await fetch(rpcUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              method: 'eth_gasPrice',
              params: [],
              id: 1
            })
          });

          if (!response.ok) continue;

          const data = await response.json();
          if (data.result) {
            const gasPriceWei = parseInt(data.result, 16);
            const gasPriceGwei = gasPriceWei / 1e9;

            return {
              chainId,
              gasPrice: gasPriceGwei,
              timestamp: Date.now(),
              loading: false,
              error: null
            };
          }
        } catch {
          continue;
        }
      }

      return { chainId, gasPrice: 0, timestamp: Date.now(), loading: false, error: 'All RPCs failed' };
    } catch (err) {
      return { chainId, gasPrice: 0, timestamp: Date.now(), loading: false, error: 'Fetch failed' };
    }
  }, []);

  // Fetch gas prices for all enabled chains
  const refreshMultiChainGas = useCallback(async () => {
    setIsLoading(true);

    // Set loading state for all chains
    const loadingState: Record<number, MultiChainGas> = {};
    enabledChains.forEach(chain => {
      loadingState[chain.id] = {
        chainId: chain.id,
        gasPrice: multiChainGas[chain.id]?.gasPrice || 0,
        timestamp: Date.now(),
        loading: true,
        error: null
      };
    });
    setMultiChainGas(loadingState);

    // Fetch all chains in parallel
    const results = await Promise.all(
      enabledChains.map(chain => fetchChainGas(chain.id))
    );

    // Update state with results
    const newState: Record<number, MultiChainGas> = {};
    results.forEach(result => {
      newState[result.chainId] = result;
    });
    setMultiChainGas(newState);
    setIsLoading(false);
  }, [enabledChains, fetchChainGas, multiChainGas]);

  // Calculate best chain for transaction
  const bestChainForTx = React.useMemo(() => {
    const validGas = Object.values(multiChainGas).filter(g => !g.error && g.gasPrice > 0);
    if (validGas.length === 0) return null;

    // Find cheapest chain
    const cheapest = validGas.reduce((min, current) =>
      current.gasPrice < min.gasPrice ? current : min
    );

    // Calculate savings vs most expensive
    const mostExpensive = validGas.reduce((max, current) =>
      current.gasPrice > max.gasPrice ? current : max
    );

    const savingsPercent = mostExpensive.gasPrice > 0
      ? ((mostExpensive.gasPrice - cheapest.gasPrice) / mostExpensive.gasPrice) * 100
      : 0;

    const chain = SUPPORTED_CHAINS[cheapest.chainId];

    return {
      chainId: cheapest.chainId,
      savings: savingsPercent,
      reason: `${chain?.name || 'Unknown'} has the lowest gas (${cheapest.gasPrice.toFixed(4)} gwei)`
    };
  }, [multiChainGas]);

  // Initial fetch
  useEffect(() => {
    refreshMultiChainGas();

    // Refresh every 30 seconds
    const interval = setInterval(refreshMultiChainGas, 30000);
    return () => clearInterval(interval);
  }, []);

  const value: ChainContextType = {
    selectedChainId,
    selectedChain,
    setSelectedChainId,
    enabledChains,
    multiChainGas,
    refreshMultiChainGas,
    bestChainForTx,
    isLoading
  };

  return (
    <ChainContext.Provider value={value}>
      {children}
    </ChainContext.Provider>
  );
};

export const useChain = (): ChainContextType => {
  const context = useContext(ChainContext);
  if (!context) {
    throw new Error('useChain must be used within a ChainProvider');
  }
  return context;
};

// Hook for getting gas price of selected chain
export const useSelectedChainGas = () => {
  const { selectedChainId, multiChainGas } = useChain();
  return multiChainGas[selectedChainId];
};

// Hook for comparing chains
export const useChainComparison = () => {
  const { multiChainGas, enabledChains } = useChain();

  return enabledChains
    .map(chain => ({
      chain,
      gas: multiChainGas[chain.id]
    }))
    .filter(item => item.gas && !item.gas.error)
    .sort((a, b) => (a.gas?.gasPrice || 0) - (b.gas?.gasPrice || 0));
};
