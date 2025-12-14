/**
 * Base blockchain integration using client-side JavaScript
 * Required for Coinbase x Queen Mary hackathon
 */

// Base network configuration
export const BASE_MAINNET = {
  chainId: 8453,
  chainIdHex: '0x2105',
  chainName: 'Base',
  nativeCurrency: {
    name: 'Ethereum',
    symbol: 'ETH',
    decimals: 18,
  },
  rpcUrls: ['https://mainnet.base.org'],
  blockExplorerUrls: ['https://basescan.org'],
};

export const BASE_TESTNET = {
  chainId: 84532,
  chainIdHex: '0x14a34',
  chainName: 'Base Sepolia',
  nativeCurrency: {
    name: 'Ethereum',
    symbol: 'ETH',
    decimals: 18,
  },
  rpcUrls: ['https://sepolia.base.org'],
  blockExplorerUrls: ['https://sepolia.basescan.org'],
};

// Window.ethereum type is defined in wallet.ts

/**
 * Connect to Base network using client-side JavaScript
 */
export async function connectToBase() {
  if (!window.ethereum) {
    throw new Error('No wallet detected. Please install MetaMask or Coinbase Wallet.');
  }

  try {
    // Request account access
    const accounts = await window.ethereum.request({
      method: 'eth_requestAccounts',
    });

    // Switch to Base network
    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: BASE_MAINNET.chainIdHex }],
      });
    } catch (switchError: any) {
      // Network not added, add it
      if (switchError.code === 4902) {
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [BASE_MAINNET],
        });
      } else {
        throw switchError;
      }
    }

    // Get current chain ID
    const chainId = await window.ethereum.request({
      method: 'eth_chainId',
    });

    return {
      address: accounts[0],
      chainId: parseInt(chainId, 16),
    };
  } catch (error) {
    console.error('Error connecting to Base:', error);
    throw error;
  }
}

/**
 * Get current gas price from Base network
 */
export async function getBaseGasPrice() {
  try {
    const response = await fetch('https://mainnet.base.org', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'eth_gasPrice',
        params: [],
        id: 1,
      }),
    });

    const data = await response.json();
    const gasPriceWei = BigInt(data.result);
    const gasPriceGwei = Number(gasPriceWei) / 1e9;

    return {
      gasPrice: gasPriceWei.toString(),
      gasPriceGwei: gasPriceGwei.toFixed(6),
    };
  } catch (error) {
    console.error('Error getting Base gas price:', error);
    throw error;
  }
}

/**
 * Get Base network information
 */
export async function getBaseNetworkInfo() {
  try {
    const [blockResponse, chainResponse] = await Promise.all([
      fetch('https://mainnet.base.org', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'eth_blockNumber',
          params: [],
          id: 1,
        }),
      }),
      fetch('https://mainnet.base.org', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'eth_chainId',
          params: [],
          id: 1,
        }),
      }),
    ]);

    const [blockData, chainData] = await Promise.all([
      blockResponse.json(),
      chainResponse.json(),
    ]);

    return {
      blockNumber: parseInt(blockData.result, 16),
      chainId: parseInt(chainData.result, 16),
      name: 'Base Mainnet',
    };
  } catch (error) {
    console.error('Error getting Base network info:', error);
    throw error;
  }
}

/**
 * Monitor Base network for gas price changes
 */
export function subscribeToBaseGasUpdates(callback: (gasPrice: string) => void) {
  // Poll every 12 seconds (Base block time)
  const interval = setInterval(async () => {
    try {
      const gasData = await getBaseGasPrice();
      callback(gasData.gasPriceGwei);
    } catch (error) {
      console.error('Error fetching gas price:', error);
    }
  }, 12000);

  // Return cleanup function
  return () => clearInterval(interval);
}

/**
 * Estimate transaction cost on Base
 */
export async function estimateTransactionCost(
  to: string,
  data: string = '0x',
  value: string = '0'
) {
  try {
    // Estimate gas
    const gasResponse = await fetch('https://mainnet.base.org', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'eth_estimateGas',
        params: [{ to, data, value }],
        id: 1,
      }),
    });

    const gasData = await gasResponse.json();
    const gasLimit = BigInt(gasData.result);

    // Get gas price
    const gasPriceData = await getBaseGasPrice();
    const gasPrice = BigInt(gasPriceData.gasPrice);

    // Calculate total cost
    const totalCostWei = gasLimit * gasPrice;
    const totalCostEth = Number(totalCostWei) / 1e18;

    return {
      gasLimit: gasLimit.toString(),
      gasPrice: gasPrice.toString(),
      gasPriceGwei: gasPriceData.gasPriceGwei,
      totalCostWei: totalCostWei.toString(),
      totalCostEth: totalCostEth.toFixed(6),
    };
  } catch (error) {
    console.error('Error estimating transaction cost:', error);
    throw error;
  }
}

