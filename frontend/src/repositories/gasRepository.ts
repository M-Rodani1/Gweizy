import { fetchCurrentGas, fetchPredictions } from '../api/gasApi';
import { CurrentGasData, PredictionsResponse } from '../../types';

export interface GasRepository {
  getCurrentGas: () => Promise<CurrentGasData>;
  getPredictions: () => Promise<PredictionsResponse>;
}

export const createGasRepository = (deps: {
  fetchCurrentGas: () => Promise<CurrentGasData>;
  fetchGasPredictions: () => Promise<PredictionsResponse>;
} = { fetchCurrentGas, fetchGasPredictions: fetchPredictions }): GasRepository => ({
  getCurrentGas: () => deps.fetchCurrentGas(),
  getPredictions: () => deps.fetchGasPredictions()
});

export const gasRepository = createGasRepository();
