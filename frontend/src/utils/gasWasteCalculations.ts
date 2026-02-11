export type GasWasteTimePeriod = 'week' | 'month' | '3months';

export interface GasWasteInputs {
  gasPrices: number[];
  gasLimit: number;
  transactionsPerWeek: number;
  timePeriod: GasWasteTimePeriod;
  ethPrice: number;
}

export interface GasWasteResult {
  avgGasPaid: number;
  optimizedGasCost: number;
  waste: number;
  wastePercent: number;
  annualWaste: number;
}

export const EMPTY_GAS_WASTE_RESULT: GasWasteResult = {
  avgGasPaid: 0,
  optimizedGasCost: 0,
  waste: 0,
  wastePercent: 0,
  annualWaste: 0,
};

export function calculateGasWaste(inputs: GasWasteInputs): GasWasteResult {
  const { gasPrices, gasLimit, transactionsPerWeek, timePeriod, ethPrice } = inputs;

  if (!gasPrices || gasPrices.length === 0) {
    return { ...EMPTY_GAS_WASTE_RESULT };
  }

  const sanitizedPrices = gasPrices.filter((price) => Number.isFinite(price) && price > 0);
  if (sanitizedPrices.length === 0) {
    return { ...EMPTY_GAS_WASTE_RESULT };
  }

  const avgGasPrice = sanitizedPrices.reduce((sum, value) => sum + value, 0) / sanitizedPrices.length;
  const sortedPrices = [...sanitizedPrices].sort((a, b) => a - b);
  const optimalIndex = Math.floor(sortedPrices.length * 0.2);
  const optimalGasPrice = sortedPrices[optimalIndex] ?? sortedPrices[0];

  const costPerTx = (gasPrice: number) => {
    const ethCost = (gasPrice * gasLimit) / 1e9;
    return ethCost * ethPrice;
  };

  const avgCostPerTx = costPerTx(avgGasPrice);
  const optimalCostPerTx = costPerTx(optimalGasPrice);

  const days = timePeriod === 'week' ? 7 : timePeriod === 'month' ? 30 : 90;
  const transactionsInPeriod = (transactionsPerWeek / 7) * days;

  const totalPaid = avgCostPerTx * transactionsInPeriod;
  const totalOptimal = optimalCostPerTx * transactionsInPeriod;
  const waste = totalPaid - totalOptimal;
  const wastePercent = totalPaid > 0 ? (waste / totalPaid) * 100 : 0;

  const annualTransactions = transactionsPerWeek * 52;
  const annualPaid = avgCostPerTx * annualTransactions;
  const annualOptimal = optimalCostPerTx * annualTransactions;
  const annualWaste = annualPaid - annualOptimal;

  return {
    avgGasPaid: totalPaid,
    optimizedGasCost: totalOptimal,
    waste,
    wastePercent,
    annualWaste,
  };
}
