import { useState, useEffect, useCallback } from 'react';

interface UseEthPriceReturn {
  ethPrice: number;
  priceChange24h: number;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refetch: () => Promise<void>;
}

// ETH price endpoint - proxied through backend to avoid CORS
const ETH_PRICE_API = 'https://basegasfeesml-production.up.railway.app/api/eth-price';

// Cache to prevent excessive API calls
let priceCache: {
  price: number;
  change: number;
  timestamp: number;
} | null = null;

const CACHE_DURATION = 60 * 1000; // 1 minute cache

export function useEthPrice(refreshInterval = 60000): UseEthPriceReturn {
  const [ethPrice, setEthPrice] = useState<number>(3500); // Default fallback
  const [priceChange24h, setPriceChange24h] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchEthPrice = useCallback(async () => {
    // Check cache first
    if (priceCache && Date.now() - priceCache.timestamp < CACHE_DURATION) {
      setEthPrice(priceCache.price);
      setPriceChange24h(priceCache.change);
      setLastUpdated(new Date(priceCache.timestamp));
      setLoading(false);
      return;
    }

    try {
      setError(null);

      const response = await fetch(ETH_PRICE_API, {
        headers: {
          'Accept': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch ETH price');
      }

      const data = await response.json();

      if (data.ethereum) {
        const price = data.ethereum.usd;
        const change = data.ethereum.usd_24h_change || 0;

        // Update cache
        priceCache = {
          price,
          change,
          timestamp: Date.now()
        };

        setEthPrice(price);
        setPriceChange24h(change);
        setLastUpdated(new Date());
      }
    } catch (err) {
      console.error('Error fetching ETH price:', err);
      setError('Failed to fetch ETH price');

      // Use cached price if available, otherwise use fallback
      if (priceCache) {
        setEthPrice(priceCache.price);
        setPriceChange24h(priceCache.change);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchEthPrice();

    // Set up interval for periodic updates
    const interval = setInterval(fetchEthPrice, refreshInterval);

    return () => clearInterval(interval);
  }, [fetchEthPrice, refreshInterval]);

  return {
    ethPrice,
    priceChange24h,
    loading,
    error,
    lastUpdated,
    refetch: fetchEthPrice
  };
}

// Standalone function to get ETH price (for use outside hooks)
export async function getEthPrice(): Promise<number> {
  // Check cache first
  if (priceCache && Date.now() - priceCache.timestamp < CACHE_DURATION) {
    return priceCache.price;
  }

  try {
    const response = await fetch(ETH_PRICE_API);
    const data = await response.json();

    if (data.ethereum) {
      priceCache = {
        price: data.ethereum.usd,
        change: data.ethereum.usd_24h_change || 0,
        timestamp: Date.now()
      };
      return data.ethereum.usd;
    }
  } catch (error) {
    console.error('Error fetching ETH price:', error);
  }

  // Return cached or fallback
  return priceCache?.price || 3500;
}

// Format USD price
export function formatUsdPrice(price: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(price);
}

// Format price change percentage
export function formatPriceChange(change: number): string {
  const sign = change >= 0 ? '+' : '';
  return `${sign}${change.toFixed(2)}%`;
}
