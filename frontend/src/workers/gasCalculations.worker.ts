import { calculateGasWaste } from '../utils/gasWasteCalculations';
import type { GasWasteInputs, GasWasteResult } from '../utils/gasWasteCalculations';

type WorkerRequest = {
  id: number;
  type: 'calculateGasWaste';
  payload: GasWasteInputs;
};

type WorkerResponse = {
  id: number;
  result?: GasWasteResult;
  error?: string;
};

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const { id, type, payload } = event.data;

  if (type !== 'calculateGasWaste') {
    const response: WorkerResponse = { id, error: 'Unsupported worker message type' };
    self.postMessage(response);
    return;
  }

  try {
    const result = calculateGasWaste(payload);
    const response: WorkerResponse = { id, result };
    self.postMessage(response);
  } catch (error) {
    const response: WorkerResponse = {
      id,
      error: error instanceof Error ? error.message : 'Failed to calculate gas waste',
    };
    self.postMessage(response);
  }
};
