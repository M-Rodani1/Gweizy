/**
 * Fetch live gas prices directly from Base network RPC
 */

// Try multiple RPC endpoints (fallback if one is rate-limited)
const BASE_RPC_URLS = [
  'https://mainnet.base.org',
  'https://base.llamarpc.com',
  'https://base-rpc.publicnode.com'
];

let currentRpcIndex = 0;

function getBaseRPC(): string {
  return BASE_RPC_URLS[currentRpcIndex % BASE_RPC_URLS.length];
}

function rotateRPC(): void {
  currentRpcIndex++;
  console.log(`🔄 Switching to RPC: ${getBaseRPC()}`);
}

interface BlockData {
  baseFeePerGas: string;
  timestamp: string;
}

/**
 * Convert hex to decimal gwei
 */
function hexToGwei(hex: string): number {
  const wei = parseInt(hex, 16);
  return wei / 1e9; // Convert wei to gwei
}

/**
 * Fetch current Base gas price
 */
export async function fetchLiveBaseGas(): Promise<{
  baseFee: number;
  gwei: number;
  timestamp: number;
}> {
  try {
    const response = await fetch(getBaseRPC(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'eth_getBlockByNumber',
        params: ['latest', false],
        id: 1
      })
    });

    const data = await response.json();
    const block: BlockData = data.result;

    const baseFee = hexToGwei(block.baseFeePerGas);
    const timestamp = parseInt(block.timestamp, 16);

    return {
      baseFee,
      gwei: baseFee, // For Base, gas price ≈ base fee (L2 has minimal priority fees)
      timestamp
    };
  } catch (error) {
    console.error('Failed to fetch live Base gas:', error);
    throw error;
  }
}

/**
 * Fetch multiple historical blocks to build time series data
 */
export async function fetchHistoricalBaseGas(hoursBack: number = 168): Promise<Array<{
  time: string;
  gwei: number;
  baseFee: number;
  timestamp: number;
}>> {
  try {
    // Get latest block number
    const latestBlockResponse = await fetch(getBaseRPC(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'eth_blockNumber',
        id: 1
      })
    });

    const latestBlockData = await latestBlockResponse.json();
    const latestBlockNum = parseInt(latestBlockData.result, 16);

    // Base network produces ~1 block every 2 seconds
    // So for X hours: blocks = X * 60 * 60 / 2
    const blocksPerHour = (60 * 60) / 2;
    const totalBlocks = Math.floor(hoursBack * blocksPerHour);

    // Sample every ~120 blocks to get reasonable data points (~1 every 4 minutes)
    // Reduced from 30 to avoid rate limiting
    const sampleInterval = 120;
    const numSamples = Math.min(Math.floor(totalBlocks / sampleInterval), 200); // Reduced from 500 to 200

    console.log(`📊 Fetching ${numSamples} historical blocks from Base network (optimized for rate limits)...`);

    // Fetch blocks in parallel (in smaller batches to avoid rate limiting)
    const batchSize = 10; // Reduced from 20 to 10
    const historicalData: Array<{
      time: string;
      gwei: number;
      baseFee: number;
      timestamp: number;
    }> = [];

    for (let i = 0; i < numSamples; i += batchSize) {
      const batchPromises = [];

      for (let j = 0; j < batchSize && (i + j) < numSamples; j++) {
        const blockOffset = (i + j) * sampleInterval;
        const blockNum = latestBlockNum - blockOffset;
        const blockHex = '0x' + blockNum.toString(16);

        batchPromises.push(
          fetch(getBaseRPC(), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              method: 'eth_getBlockByNumber',
              params: [blockHex, false],
              id: i + j
            })
          }).then(res => res.json())
        );
      }

      const batchResults = await Promise.all(batchPromises);

      for (const result of batchResults) {
        if (result.result && result.result.baseFeePerGas) {
          const block: BlockData = result.result;
          const baseFee = hexToGwei(block.baseFeePerGas);
          const timestamp = parseInt(block.timestamp, 16);
          const date = new Date(timestamp * 1000);

          historicalData.push({
            time: date.toISOString(),
            gwei: baseFee,
            baseFee,
            timestamp
          });
        }
      }

      // Longer delay between batches to avoid rate limiting
      if (i + batchSize < numSamples) {
        await new Promise(resolve => setTimeout(resolve, 500)); // Increased from 100ms to 500ms
      }
    }

    console.log(`✅ Fetched ${historicalData.length} historical data points from Base network`);

    // Sort by timestamp (oldest first)
    return historicalData.sort((a, b) => a.timestamp - b.timestamp);

  } catch (error) {
    console.error('Failed to fetch historical Base gas:', error);
    throw error;
  }
}
