// Wallet connection utilities for MetaMask/Coinbase Wallet

declare global {
  interface Window {
    ethereum?: {
      request: (args: { method: string; params?: any[] }) => Promise<any>;
      isMetaMask?: boolean;
      isCoinbaseWallet?: boolean;
      on: (event: string, handler: (accounts: string[]) => void) => void;
      removeListener: (event: string, handler: (accounts: string[]) => void) => void;
    };
  }
}

const BASE_CHAIN_ID = '0x2105'; // Base mainnet = 8453 decimal
const BASE_CHAIN_ID_DECIMAL = 8453;

export interface WalletInfo {
  address: string;
  chainId: number;
  isConnected: boolean;
}

/**
 * Check if a wallet is installed
 */
export function isWalletInstalled(): boolean {
  return typeof window !== 'undefined' && !!window.ethereum;
}

/**
 * Get current wallet address
 */
export async function getCurrentAccount(): Promise<string | null> {
  if (!window.ethereum) return null;
  
  try {
    const accounts = await window.ethereum.request({ method: 'eth_accounts' });
    return accounts.length > 0 ? accounts[0] : null;
  } catch (error) {
    console.error('Error getting current account:', error);
    return null;
  }
}

/**
 * Get current chain ID
 */
export async function getCurrentChainId(): Promise<number | null> {
  if (!window.ethereum) return null;
  
  try {
    const chainId = await window.ethereum.request({ method: 'eth_chainId' });
    return parseInt(chainId, 16);
  } catch (error) {
    console.error('Error getting chain ID:', error);
    return null;
  }
}

/**
 * Connect wallet and switch to Base network
 */
export async function connectWallet(): Promise<string> {
  if (!window.ethereum) {
    throw new Error('No wallet detected. Please install MetaMask or Coinbase Wallet.');
  }

  // Request account access
  const accounts = await window.ethereum.request({ 
    method: 'eth_requestAccounts' 
  });

  if (!accounts || accounts.length === 0) {
    throw new Error('No accounts found. Please unlock your wallet.');
  }

  const address = accounts[0];

  // Switch to Base network
  try {
    await window.ethereum.request({
      method: 'wallet_switchEthereumChain',
      params: [{ chainId: BASE_CHAIN_ID }], // Base = 8453 = 0x2105
    });
  } catch (switchError: any) {
    // If Base network not added, add it
    if (switchError.code === 4902) {
      await window.ethereum.request({
        method: 'wallet_addEthereumChain',
        params: [{
          chainId: BASE_CHAIN_ID,
          chainName: 'Base',
          nativeCurrency: {
            name: 'Ether',
            symbol: 'ETH',
            decimals: 18
          },
          rpcUrls: ['https://mainnet.base.org'],
          blockExplorerUrls: ['https://basescan.org']
        }]
      });
    } else {
      throw switchError;
    }
  }

  return address;
}

/**
 * Disconnect wallet
 */
export async function disconnectWallet(): Promise<void> {
  // MetaMask doesn't have a disconnect method, but we can clear local state
  // The wallet will remain connected until user disconnects manually
  return Promise.resolve();
}

/**
 * Format address for display
 */
export function formatAddress(address: string): string {
  if (!address) return '';
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

/**
 * Copy address to clipboard
 */
export async function copyToClipboard(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch (error) {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.opacity = '0';
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
  }
}

/**
 * Get BaseScan URL for address
 */
export function getBaseScanAddressUrl(address: string): string {
  return `https://basescan.org/address/${address}`;
}

/**
 * Listen for account changes
 */
export function onAccountsChanged(callback: (accounts: string[]) => void): () => void {
  if (!window.ethereum) {
    return () => {};
  }

  window.ethereum.on('accountsChanged', callback);

  return () => {
    if (window.ethereum) {
      window.ethereum.removeListener('accountsChanged', callback);
    }
  };
}

/**
 * Listen for chain changes
 */
export function onChainChanged(callback: (chainId: string) => void): () => void {
  if (!window.ethereum) {
    return () => {};
  }

  window.ethereum.on('chainChanged', callback);

  return () => {
    if (window.ethereum) {
      window.ethereum.removeListener('chainChanged', callback);
    }
  };
}

