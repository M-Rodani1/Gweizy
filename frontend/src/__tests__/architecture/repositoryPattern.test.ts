import { describe, it, expect, vi } from 'vitest';
import { createGasRepository } from '../../repositories/gasRepository';

describe('repository pattern for data', () => {
  it('delegates data access to underlying api functions', async () => {
    const fetchCurrentGas = vi.fn().mockResolvedValue({ current: 1 });
    const fetchGasPredictions = vi.fn().mockResolvedValue({ predictions: [] });

    const repo = createGasRepository({ fetchCurrentGas, fetchGasPredictions } as any);

    await repo.getCurrentGas();
    await repo.getPredictions();

    expect(fetchCurrentGas).toHaveBeenCalled();
    expect(fetchGasPredictions).toHaveBeenCalled();
  });
});
