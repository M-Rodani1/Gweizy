import React, { createContext, useContext, useState, useEffect, useCallback, useMemo, useRef, ReactNode } from 'react';
import { ChainConfig, SUPPORTED_CHAINS, DEFAULT_CHAIN_ID, getChainById, getEnabledChains } from '../config/chains';
import { REFRESH_INTERVALS } from '../constants';
import { safeGetLocalStorageItem, safeSetLocalStorageItem } from '../utils/safeStorage';

interface MultiChainGas {
  chainId: number;
  gasPrice: number;
  timestamp: number;
  loading: boolean;
  error: string | null;
  source?: 'live' | 'cached';
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
const GAS_CACHE_PREFIX = 'gweizy_chain_gas_v1:';
const GAS_CACHE_TTL_MS = 60 * 1000;
const RPC_TIMEOUT_MS = 5000;
const ENDPOINT_BASE_BACKOFF_MS = 60 * 1000;
const ENDPOINT_MAX_BACKOFF_MS = 15 * 60 * 1000;

const endpointBackoffMs = new Map<string, number>();
const endpointBlockedUntilMs = new Map<string, number>();

const getGasCacheKey = (chainId: number): string => `${GAS_CACHE_PREFIX}${chainId}`;

const readCachedGas = (chainId: number): MultiChainGas | null => {
  if (typeof window === 'undefined') return null;
  const key = getGasCacheKey(chainId);
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { gasPrice: number; timestamp: number };
    if (Date.now() - parsed.timestamp > GAS_CACHE_TTL_MS) {
      localStorage.removeItem(key);
      return null;
    }
    return {
      chainId,
      gasPrice: parsed.gasPrice,
      timestamp: parsed.timestamp,
      loading: false,
      error: null,
      source: 'cached'
    };
  } catch {
    return null;
  }
};

const writeCachedGas = (chainId: number, gasPrice: number, timestamp: number) => {
  if (typeof window === 'undefined') return;
  try {
    const key = getGasCacheKey(chainId);
    localStorage.setItem(key, JSON.stringify({ gasPrice, timestamp }));
  } catch {
    // Ignore cache failures.
  }
};

const getRetryAfterMs = (response: Response): number | null => {
  const retryAfterHeader = response.headers.get('retry-after');
  if (!retryAfterHeader) return null;

  const retryAfterSeconds = Number.parseInt(retryAfterHeader, 10);
  if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0) {
    return retryAfterSeconds * 1000;
  }

  const retryAt = Date.parse(retryAfterHeader);
  if (!Number.isNaN(retryAt)) {
    const ms = retryAt - Date.now();
    return ms > 0 ? ms : null;
  }

  return null;
};

const isEndpointCoolingDown = (rpcUrl: string): boolean => {
  const blockedUntil = endpointBlockedUntilMs.get(rpcUrl) ?? 0;
  return blockedUntil > Date.now();
};

const markEndpointFailure = (rpcUrl: string, retryAfterMs?: number) => {
  const previousBackoff = endpointBackoffMs.get(rpcUrl) ?? ENDPOINT_BASE_BACKOFF_MS;
  const computedBackoff = Math.min(previousBackoff * 2, ENDPOINT_MAX_BACKOFF_MS);
  const nextBackoff = retryAfterMs
    ? Math.min(Math.max(retryAfterMs, ENDPOINT_BASE_BACKOFF_MS), ENDPOINT_MAX_BACKOFF_MS)
    : computedBackoff;

  endpointBackoffMs.set(rpcUrl, nextBackoff);
  endpointBlockedUntilMs.set(rpcUrl, Date.now() + nextBackoff);
};

const markEndpointSuccess = (rpcUrl: string) => {
  endpointBackoffMs.delete(rpcUrl);
  endpointBlockedUntilMs.delete(rpcUrl);
};

export const ChainProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Load saved chain from localStorage or use default
  const [selectedChainId, setSelectedChainIdState] = useState<number>(() => {
    const saved = safeGetLocalStorageItem(STORAGE_KEY);
    if (saved) {
      const chainId = parseInt(saved, 10);
      if (SUPPORTED_CHAINS[chainId]?.enabled) {
        return chainId;
      }
    }
    return DEFAULT_CHAIN_ID;
  });

  const [multiChainGas, setMultiChainGas] = useState<Record<number, MultiChainGas>>({});
  const [isLoading, setIsLoading] = useState(true);
  const refreshInFlightRef = useRef<Promise<void> | null>(null);

  const enabledChains = useMemo(() => getEnabledChains(), []);
  const selectedChain = getChainById(selectedChainId) || SUPPORTED_CHAINS[DEFAULT_CHAIN_ID];

  // Persist selected chain
  const setSelectedChainId = useCallback((chainId: number) => {
    if (SUPPORTED_CHAINS[chainId]?.enabled) {
      setSelectedChainIdState(chainId);
      safeSetLocalStorageItem(STORAGE_KEY, chainId.toString());
    }
  }, []);

  // Fetch gas price for a single chain
  const fetchChainGas = useCallback(async (chainId: number): Promise<MultiChainGas> => {
    const chain = SUPPORTED_CHAINS[chainId];
    if (!chain) {
      return { chainId, gasPrice: 0, timestamp: Date.now(), loading: false, error: 'Unknown chain' };
    }

    try {
      const availableRpcUrls = chain.rpcUrls.filter(rpcUrl => !isEndpointCoolingDown(rpcUrl));
      const rpcUrlsToTry = availableRpcUrls.length > 0 ? availableRpcUrls : chain.rpcUrls;

      // Try each RPC until one works
      for (const rpcUrl of rpcUrlsToTry) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), RPC_TIMEOUT_MS);
        try {
          const response = await fetch(rpcUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              method: 'eth_gasPrice',
              params: [],
              id: 1
            }),
            signal: controller.signal
          });

          if (!response.ok) {
            markEndpointFailure(rpcUrl, getRetryAfterMs(response) ?? undefined);
            continue;
          }

          const data = await response.json();
          if (data.result) {
            const gasPriceWei = parseInt(data.result, 16);
            const gasPriceGwei = gasPriceWei / 1e9;
            const timestamp = Date.now();
            markEndpointSuccess(rpcUrl);
            writeCachedGas(chainId, gasPriceGwei, timestamp);

            return {
              chainId,
              gasPrice: gasPriceGwei,
              timestamp,
              loading: false,
              error: null,
              source: 'live'
            };
          }
          markEndpointFailure(rpcUrl);
        } catch {
          markEndpointFailure(rpcUrl);
          continue;
        } finally {
          clearTimeout(timeoutId);
        }
      }

      const cached = readCachedGas(chainId);
      if (cached) {
        return cached;
      }
      return { chainId, gasPrice: 0, timestamp: Date.now(), loading: false, error: 'All RPCs failed' };
    } catch {
      const cached = readCachedGas(chainId);
      if (cached) {
        return cached;
      }
      return { chainId, gasPrice: 0, timestamp: Date.now(), loading: false, error: 'Fetch failed' };
    }
  }, []);

  // Fetch gas prices for all enabled chains
  const refreshMultiChainGas = useCallback(async () => {
    if (refreshInFlightRef.current) {
      await refreshInFlightRef.current;
      return;
    }

    const refreshTask = (async () => {
      setIsLoading(true);

      try {
        // Set loading state for all chains
        setMultiChainGas(previousState => {
          const loadingState: Record<number, MultiChainGas> = {};
          enabledChains.forEach(chain => {
            loadingState[chain.id] = {
              chainId: chain.id,
              gasPrice: previousState[chain.id]?.gasPrice || 0,
              timestamp: Date.now(),
              loading: true,
              error: null,
              source: previousState[chain.id]?.source
            };
          });
          return loadingState;
        });

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
      } finally {
        setIsLoading(false);
      }
    })();

    refreshInFlightRef.current = refreshTask;
    try {
      await refreshTask;
    } finally {
      refreshInFlightRef.current = null;
    }
  }, [enabledChains, fetchChainGas]);

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
    void refreshMultiChainGas();

    const refreshIfVisible = () => {
      if (typeof document === 'undefined' || document.visibilityState === 'visible') {
        void refreshMultiChainGas();
      }
    };

    const interval = setInterval(refreshIfVisible, REFRESH_INTERVALS.GAS_DATA);
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        void refreshMultiChainGas();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [refreshMultiChainGas]);

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
