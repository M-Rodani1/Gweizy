// Multi-chain configuration for gas prediction

export interface ChainConfig {
  id: number;
  name: string;
  shortName: string;
  nativeCurrency: {
    name: string;
    symbol: string;
    decimals: number;
  };
  rpcUrls: string[];
  blockExplorer: string;
  icon: string;
  color: string;
  isL2: boolean;
  enabled: boolean;
  // Gas-specific config
  gasUnit: 'gwei' | 'wei';
  avgBlockTime: number; // seconds
  supportsEIP1559: boolean;
}

export const SUPPORTED_CHAINS: Record<number, ChainConfig> = {
  // Base (Primary - fully supported)
  8453: {
    id: 8453,
    name: 'Base',
    shortName: 'BASE',
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: [
      'https://mainnet.base.org',
      'https://base.llamarpc.com',
      'https://base.drpc.org'
    ],
    blockExplorer: 'https://basescan.org',
    icon: '🔵',
    color: '#0052FF',
    isL2: true,
    enabled: true,
    gasUnit: 'gwei',
    avgBlockTime: 2,
    supportsEIP1559: true
  },

  // Ethereum Mainnet
  1: {
    id: 1,
    name: 'Ethereum',
    shortName: 'ETH',
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: [
      'https://eth.llamarpc.com',
      'https://ethereum.publicnode.com',
      'https://rpc.ankr.com/eth'
    ],
    blockExplorer: 'https://etherscan.io',
    icon: '⟠',
    color: '#627EEA',
    isL2: false,
    enabled: true,
    gasUnit: 'gwei',
    avgBlockTime: 12,
    supportsEIP1559: true
  },

  // Arbitrum One
  42161: {
    id: 42161,
    name: 'Arbitrum',
    shortName: 'ARB',
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: [
      'https://arb1.arbitrum.io/rpc',
      'https://arbitrum.llamarpc.com',
      'https://arbitrum-one.publicnode.com'
    ],
    blockExplorer: 'https://arbiscan.io',
    icon: '🔷',
    color: '#28A0F0',
    isL2: true,
    enabled: true,
    gasUnit: 'gwei',
    avgBlockTime: 0.25,
    supportsEIP1559: true
  },

  // Optimism
  10: {
    id: 10,
    name: 'Optimism',
    shortName: 'OP',
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: [
      'https://mainnet.optimism.io',
      'https://optimism.llamarpc.com',
      'https://optimism.publicnode.com'
    ],
    blockExplorer: 'https://optimistic.etherscan.io',
    icon: '🔴',
    color: '#FF0420',
    isL2: true,
    enabled: true,
    gasUnit: 'gwei',
    avgBlockTime: 2,
    supportsEIP1559: true
  },

  // Polygon
  137: {
    id: 137,
    name: 'Polygon',
    shortName: 'MATIC',
    nativeCurrency: { name: 'MATIC', symbol: 'MATIC', decimals: 18 },
    rpcUrls: [
      'https://polygon-rpc.com',
      'https://polygon.llamarpc.com',
      'https://polygon-bor.publicnode.com'
    ],
    blockExplorer: 'https://polygonscan.com',
    icon: '💜',
    color: '#8247E5',
    isL2: true,
    enabled: true,
    gasUnit: 'gwei',
    avgBlockTime: 2,
    supportsEIP1559: true
  }
};

// Default chain
export const DEFAULT_CHAIN_ID = 8453; // Base

// Get enabled chains only
export const getEnabledChains = (): ChainConfig[] => {
  return Object.values(SUPPORTED_CHAINS).filter(chain => chain.enabled);
};

// Get chain by ID
export const getChainById = (chainId: number): ChainConfig | undefined => {
  return SUPPORTED_CHAINS[chainId];
};

// Get chain display name
export const getChainDisplayName = (chainId: number): string => {
  return SUPPORTED_CHAINS[chainId]?.name || `Chain ${chainId}`;
};

// Transaction type gas estimates (in gas units)
export const TX_GAS_ESTIMATES = {
  transfer: 21000,
  erc20Transfer: 65000,
  swap: 150000,
  nftMint: 100000,
  nftTransfer: 80000,
  contractDeploy: 500000,
  bridge: 200000,
  approve: 46000
} as const;

export type TransactionType = keyof typeof TX_GAS_ESTIMATES;
