/**
 * Zustand store for gas data state management
 * Provides global state for gas prices and predictions
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface GasState {
  currentGas: number;
  predictions: {
    '1h': number;
    '4h': number;
    '24h': number;
  };
  lastUpdated: number | null;
  setCurrentGas: (gas: number) => void;
  setPredictions: (predictions: { '1h': number; '4h': number; '24h': number }) => void;
  updateLastUpdated: () => void;
}

export const useGasStore = create<GasState>()(
  persist(
    (set) => ({
      currentGas: 0,
      predictions: { '1h': 0, '4h': 0, '24h': 0 },
      lastUpdated: null,
      setCurrentGas: (gas) => set({ currentGas: gas }),
      setPredictions: (predictions) => set({ predictions }),
      updateLastUpdated: () => set({ lastUpdated: Date.now() }),
    }),
    {
      name: 'gas-storage',
      partialize: (state) => ({
        currentGas: state.currentGas,
        predictions: state.predictions,
        lastUpdated: state.lastUpdated,
      }),
    }
  )
);
