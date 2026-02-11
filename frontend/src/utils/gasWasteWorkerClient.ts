import type { GasWasteInputs, GasWasteResult } from './gasWasteCalculations';
import { calculateGasWaste } from './gasWasteCalculations';

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

let worker: Worker | null = null;
let requestId = 0;
const pending = new Map<number, { resolve: (value: GasWasteResult) => void; reject: (reason?: Error) => void }>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('../workers/gasCalculations.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const { id, result, error } = event.data;
      const entry = pending.get(id);
      if (!entry) return;
      pending.delete(id);
      if (error) {
        entry.reject(new Error(error));
        return;
      }
      if (result) {
        entry.resolve(result);
      } else {
        entry.reject(new Error('Worker returned empty result'));
      }
    };
    worker.onerror = (event) => {
      pending.forEach(({ reject }) => reject(new Error(event.message)));
      pending.clear();
    };
  }

  return worker;
}

export function terminateGasWasteWorker(): void {
  if (worker) {
    worker.terminate();
    worker = null;
    pending.clear();
  }
}

export async function calculateGasWasteAsync(inputs: GasWasteInputs): Promise<GasWasteResult> {
  if (typeof Worker === 'undefined') {
    return calculateGasWaste(inputs);
  }

  const activeWorker = getWorker();

  return new Promise<GasWasteResult>((resolve, reject) => {
    const id = requestId++;
    pending.set(id, { resolve, reject });
    const message: WorkerRequest = { id, type: 'calculateGasWaste', payload: inputs };
    activeWorker.postMessage(message);
  });
}
