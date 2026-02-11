import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { calculateGasWaste, type GasWasteInputs } from '../../utils/gasWasteCalculations';
import { calculateGasWasteAsync } from '../../utils/gasWasteWorkerClient';

describe('Gas waste calculations', () => {
  it('computes waste metrics for a weekly period', () => {
    const inputs: GasWasteInputs = {
      gasPrices: [10, 20, 30, 40, 50],
      gasLimit: 100000,
      transactionsPerWeek: 7,
      timePeriod: 'week',
      ethPrice: 2000,
    };

    const result = calculateGasWaste(inputs);

    expect(result.avgGasPaid).toBeCloseTo(42, 5);
    expect(result.optimizedGasCost).toBeCloseTo(28, 5);
    expect(result.waste).toBeCloseTo(14, 5);
    expect(result.wastePercent).toBeCloseTo(33.3333, 3);
    expect(result.annualWaste).toBeCloseTo(728, 5);
  });

  it('falls back to synchronous calculation when Worker is unavailable', async () => {
    const originalWorker = globalThis.Worker;
    vi.stubGlobal('Worker', undefined as unknown as typeof Worker);

    const inputs: GasWasteInputs = {
      gasPrices: [15, 25, 35],
      gasLimit: 21000,
      transactionsPerWeek: 3,
      timePeriod: 'month',
      ethPrice: 2500,
    };

    const asyncResult = await calculateGasWasteAsync(inputs);
    const syncResult = calculateGasWaste(inputs);

    expect(asyncResult).toEqual(syncResult);

    vi.stubGlobal('Worker', originalWorker);
  });
});
