/**
 * Gas-specific WebSocket hook for real-time price updates.
 *
 * Provides a Socket.IO connection to the backend for receiving
 * live gas price, prediction, and mempool updates. Includes automatic
 * reconnection with exponential backoff and connection state management.
 *
 * @module hooks/useGasWebSocket
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { getApiOrigin } from '../config/api';

// ========================================
// Types
// ========================================

/**
 * Gas price update payload received from WebSocket.
 */
export interface GasPriceUpdate {
  current_gas: number;
  base_fee: number;
  priority_fee: number;
  timestamp: string;
  collection_count: number;
}

/**
 * Prediction update payload from WebSocket.
 */
export interface PredictionUpdate {
  current_price: number;
  predictions: {
    [horizon: string]: {
      price: number;
      confidence: number;
      lower_bound: number;
      upper_bound: number;
    };
  };
  timestamp: string;
}

/**
 * Mempool status update from WebSocket.
 */
export interface MempoolUpdate {
  pending_count: number;
  avg_gas_price: number;
  is_congested: boolean;
  gas_momentum: number;
  count_momentum: number;
  timestamp: string;
}

/**
 * Combined update with all real-time data.
 */
export interface CombinedGasUpdate {
  gas: GasPriceUpdate;
  predictions?: PredictionUpdate;
  mempool?: MempoolUpdate;
  timestamp: string;
}

/**
 * Configuration options for the WebSocket hook.
 */
export interface UseGasWebSocketOptions {
  /** Whether the WebSocket connection should be active (default: true) */
  enabled?: boolean;
  /** Callback fired when connection is established */
  onConnect?: () => void;
  /** Callback fired when connection is lost */
  onDisconnect?: () => void;
  /** Callback fired when a connection error occurs */
  onError?: (error: Error) => void;
  /** Callback fired when gas price updates */
  onGasPriceUpdate?: (data: GasPriceUpdate) => void;
  /** Callback fired when predictions update */
  onPredictionUpdate?: (data: PredictionUpdate) => void;
  /** Callback fired when mempool status updates */
  onMempoolUpdate?: (data: MempoolUpdate) => void;
}

/**
 * Return type for useGasWebSocket hook.
 */
export interface UseGasWebSocketReturn {
  /** The Socket.IO socket instance */
  socket: Socket | null;
  /** Current connection status */
  isConnected: boolean;
  /** Latest gas price data */
  gasPrice: GasPriceUpdate | null;
  /** Latest prediction data */
  predictions: PredictionUpdate | null;
  /** Latest mempool data */
  mempool: MempoolUpdate | null;
  /** Any connection error */
  error: Error | null;
  /** Manually disconnect */
  disconnect: () => void;
  /** Manually trigger reconnection */
  reconnect: () => void;
}

// ========================================
// Constants
// ========================================

const MAX_RECONNECT_ATTEMPTS = 5;
const INITIAL_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 5000;
const CONNECTION_TIMEOUT = 20000;

// ========================================
// Hook
// ========================================

/**
 * Hook for real-time WebSocket communication with the gas price backend.
 *
 * Establishes a Socket.IO connection that receives live gas price updates.
 * Handles automatic reconnection with exponential backoff (up to 5 attempts).
 *
 * @param options - Configuration options
 * @returns WebSocket state and controls
 *
 * @example
 * ```tsx
 * function GasDisplay() {
 *   const { isConnected, gasPrice, error } = useGasWebSocket({
 *     onConnect: () => console.log('Connected!'),
 *     onGasPriceUpdate: (data) => console.log('New price:', data.current_gas),
 *     onError: (err) => console.error('WS Error:', err),
 *   });
 *
 *   if (error) return <div>Connection error</div>;
 *   if (!isConnected) return <div>Connecting...</div>;
 *   return <div>Gas: {gasPrice?.current_gas} gwei</div>;
 * }
 * ```
 */
export function useGasWebSocket(
  options: UseGasWebSocketOptions = {}
): UseGasWebSocketReturn {
  const {
    enabled = true,
    onConnect,
    onDisconnect,
    onError,
    onGasPriceUpdate,
    onPredictionUpdate,
    onMempoolUpdate,
  } = options;

  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [gasPrice, setGasPrice] = useState<GasPriceUpdate | null>(null);
  const [predictions, setPredictions] = useState<PredictionUpdate | null>(null);
  const [mempool, setMempool] = useState<MempoolUpdate | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const reconnectAttempts = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const apiOrigin = getApiOrigin();

    // Create Socket.IO connection
    const newSocket = io(apiOrigin, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
      reconnectionDelay: INITIAL_RECONNECT_DELAY,
      reconnectionDelayMax: MAX_RECONNECT_DELAY,
      timeout: CONNECTION_TIMEOUT,
    });

    // Connection established
    newSocket.on('connect', () => {
      setIsConnected(true);
      setError(null);
      reconnectAttempts.current = 0;
      onConnect?.();
    });

    // Connection lost
    newSocket.on('disconnect', (reason: string) => {
      setIsConnected(false);
      onDisconnect?.();

      // Attempt reconnection if not intentional
      if (reason !== 'io client disconnect') {
        reconnectAttempts.current += 1;
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          const delay = Math.min(
            INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttempts.current),
            MAX_RECONNECT_DELAY
          );
          reconnectTimeoutRef.current = setTimeout(() => {
            newSocket.connect();
          }, delay);
        } else {
          const err = new Error('WebSocket connection failed after multiple attempts');
          setError(err);
          onError?.(err);
        }
      }
    });

    // Connection error
    newSocket.on('connect_error', (err: Error) => {
      console.error('WebSocket connection error:', err);
      setError(err);
      onError?.(err);
    });

    // Gas price update
    newSocket.on('gas_price_update', (data: GasPriceUpdate) => {
      setGasPrice(data);
      setError(null);
      onGasPriceUpdate?.(data);
    });

    // Prediction update
    newSocket.on('prediction_update', (data: PredictionUpdate) => {
      setPredictions(data);
      onPredictionUpdate?.(data);
    });

    // Mempool status update
    newSocket.on('mempool_update', (data: MempoolUpdate) => {
      setMempool(data);
      onMempoolUpdate?.(data);
    });

    // Combined update (gas + predictions + mempool)
    newSocket.on('combined_update', (data: CombinedGasUpdate) => {
      if (data.gas) {
        setGasPrice(data.gas);
        onGasPriceUpdate?.(data.gas);
      }
      if (data.predictions) {
        setPredictions(data.predictions);
        onPredictionUpdate?.(data.predictions);
      }
      if (data.mempool) {
        setMempool(data.mempool);
        onMempoolUpdate?.(data.mempool);
      }
      setError(null);
    });

    // Connection established confirmation
    newSocket.on('connection_established', (_data: { message: string }) => {
      // Connection confirmed by server
    });

    setSocket(newSocket);

    // Cleanup
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      newSocket.disconnect();
      setSocket(null);
      setIsConnected(false);
    };
  }, [enabled, onConnect, onDisconnect, onError, onGasPriceUpdate, onPredictionUpdate, onMempoolUpdate]);

  const disconnect = useCallback(() => {
    if (socket) {
      socket.disconnect();
    }
  }, [socket]);

  const reconnect = useCallback(() => {
    if (socket) {
      reconnectAttempts.current = 0;
      socket.connect();
    }
  }, [socket]);

  return {
    socket,
    isConnected,
    gasPrice,
    predictions,
    mempool,
    error,
    disconnect,
    reconnect,
  };
}

// Re-export with backwards compatible name
export { useGasWebSocket as useWebSocket };
