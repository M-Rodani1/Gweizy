// Wallet connection utilities for MetaMask/Coinbase Wallet
import { SUPPORTED_CHAINS } from '../config/chains';

declare global {
  interface Window {
    ethereum?: {
      request: (args: { method: string; params?: any[] }) => Promise<any>;
      isMetaMask?: boolean;
      isCoinbaseWallet?: boolean;
      on: (event: string, handler: (...args: any[]) => void) => void;
      removeListener: (event: string, handler: (...args: any[]) => void) => void;
    };
  }
}

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

const toHex = (value: bigint): string => `0x${value.toString(16)}`;

const parseUnits = (value: string, decimals: number): bigint => {
  const [whole, fraction = ''] = value.split('.');
  const padded = fraction.padEnd(decimals, '0').slice(0, decimals);
  const wholeValue = whole ? BigInt(whole) : 0n;
  const fractionValue = padded ? BigInt(padded) : 0n;
  return wholeValue * 10n ** BigInt(decimals) + fractionValue;
};

export const gweiToWeiHex = (gwei: string): string => toHex(parseUnits(gwei, 9));
export const ethToWeiHex = (eth: string): string => toHex(parseUnits(eth, 18));

const getChainParams = (chainId: number) => {
  const chain = SUPPORTED_CHAINS[chainId];
  if (!chain) return null;
  return {
    chainId: `0x${chainId.toString(16)}`,
    chainName: chain.name,
    nativeCurrency: chain.nativeCurrency,
    rpcUrls: chain.rpcUrls,
    blockExplorerUrls: [chain.blockExplorer]
  };
};

export async function switchToChain(chainId: number): Promise<void> {
  if (!window.ethereum) {
    throw new Error('No wallet detected.');
  }
  const chainParams = getChainParams(chainId);
  if (!chainParams) {
    throw new Error('Unsupported chain.');
  }

  try {
    await window.ethereum.request({
      method: 'wallet_switchEthereumChain',
      params: [{ chainId: chainParams.chainId }]
    });
  } catch (switchError: any) {
    if (switchError.code === 4902) {
      await window.ethereum.request({
        method: 'wallet_addEthereumChain',
        params: [chainParams]
      });
    } else {
      throw switchError;
    }
  }
}

/**
 * Connect wallet and optionally switch to target network
 */
export async function connectWallet(targetChainId: number = BASE_CHAIN_ID_DECIMAL): Promise<string> {
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

  // Switch to target network
  await switchToChain(targetChainId);

  return address;
}

interface SendTransactionParams {
  to: string;
  valueEth?: string;
  gasPriceGwei?: string;
  gasLimit?: string;
  data?: string;
}

export async function sendTransaction(params: SendTransactionParams): Promise<string> {
  if (!window.ethereum) {
    throw new Error('No wallet detected.');
  }
  const from = await getCurrentAccount();
  if (!from) {
    throw new Error('Wallet not connected.');
  }

  const tx: Record<string, string> = {
    from,
    to: params.to
  };

  if (params.valueEth !== undefined) {
    tx.value = ethToWeiHex(params.valueEth);
  }
  if (params.gasPriceGwei) {
    tx.gasPrice = gweiToWeiHex(params.gasPriceGwei);
  }
  if (params.gasLimit) {
    tx.gas = toHex(BigInt(params.gasLimit));
  }
  if (params.data) {
    tx.data = params.data;
  }

  const txHash = await window.ethereum.request({
    method: 'eth_sendTransaction',
    params: [tx]
  });

  return txHash as string;
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
