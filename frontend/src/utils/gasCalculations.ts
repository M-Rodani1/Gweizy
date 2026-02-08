/**
 * Gas calculation utilities for transaction cost estimation
 */

import { TX_GAS_ESTIMATES, TransactionType } from '../config/chains';

export interface GasCostEstimate {
  gasUnits: number;
  costEth: number;
  costUsd: number;
}

/**
 * Calculate estimated gas cost for a transaction type
 */
export const calculateGasCost = (
  txType: TransactionType,
  currentGasGwei: number,
  ethPrice: number
): GasCostEstimate => {
  const gasUnits = TX_GAS_ESTIMATES[txType];
  const costEth = (currentGasGwei * gasUnits) / 1e9;
  const costUsd = costEth * ethPrice;
  
  return {
    gasUnits,
    costEth,
    costUsd
  };
};

/**
 * Format gas cost for display
 */
export const formatGasCost = (costUsd: number): string => {
  if (costUsd < 0.01) return '<$0.01';
  if (costUsd < 1) return `$${costUsd.toFixed(2)}`;
  return `$${costUsd.toFixed(2)}`;
};
