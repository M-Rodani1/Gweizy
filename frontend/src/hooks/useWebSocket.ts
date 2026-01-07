/**
 * WebSocket hook for real-time gas price updates.
 *
 * Provides a Socket.IO connection to the backend for receiving
 * live gas price updates. Includes automatic reconnection with
 * exponential backoff and connection state management.
 *
 * @module hooks/useWebSocket
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { getApiOrigin } from '../config/api';

/**
 * Gas price update payload received from WebSocket.
 */
interface GasPriceUpdate {
  current_gas: number;
  base_fee: number;
  priority_fee: number;
  timestamp: string;
  collection_count: number;
}

/**
 * Configuration options for the WebSocket hook.
 */
interface UseWebSocketOptions {
  /** Whether the WebSocket connection should be active (default: true) */
  enabled?: boolean;
  /** Callback fired when connection is established */
  onConnect?: () => void;
  /** Callback fired when connection is lost */
  onDisconnect?: () => void;
  /** Callback fired when a connection error occurs */
  onError?: (error: Error) => void;
}

/**
 * Hook for real-time WebSocket communication with the gas price backend.
 *
 * Establishes a Socket.IO connection that receives live gas price updates.
 * Handles automatic reconnection with exponential backoff (up to 5 attempts).
 *
 * @param {UseWebSocketOptions} options - Configuration options
 * @param {boolean} [options.enabled=true] - Whether to enable the connection
 * @param {Function} [options.onConnect] - Callback when connected
 * @param {Function} [options.onDisconnect] - Callback when disconnected
 * @param {Function} [options.onError] - Callback when error occurs
 *
 * @returns {Object} WebSocket state and controls
 * @returns {Socket|null} returns.socket - The Socket.IO socket instance
 * @returns {boolean} returns.isConnected - Current connection status
 * @returns {GasPriceUpdate|null} returns.gasPrice - Latest gas price data
 * @returns {Error|null} returns.error - Any connection error
 * @returns {Function} returns.disconnect - Manually disconnect
 * @returns {Function} returns.reconnect - Manually trigger reconnection
 *
 * @example
 * ```tsx
 * function GasDisplay() {
 *   const { isConnected, gasPrice, error } = useWebSocket({
 *     onConnect: () => console.log('Connected!'),
 *     onError: (err) => console.error('WS Error:', err),
 *   });
 *
 *   if (error) return <div>Connection error</div>;
 *   if (!isConnected) return <div>Connecting...</div>;
 *   return <div>Gas: {gasPrice?.current_gas} gwei</div>;
 * }
 * ```
 */
export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { enabled = true, onConnect, onDisconnect, onError } = options;
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [gasPrice, setGasPrice] = useState<GasPriceUpdate | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    const apiOrigin = getApiOrigin();
    // Socket.IO handles protocol automatically, just use the origin
    const socketUrl = apiOrigin;
    
    // Create Socket.IO connection
    const newSocket = io(socketUrl, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: maxReconnectAttempts,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    });

    // Connection established
    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setError(null);
      reconnectAttempts.current = 0;
      onConnect?.();
    });

    // Connection lost
    newSocket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      setIsConnected(false);
      onDisconnect?.();
      
      // Attempt reconnection if not intentional
      if (reason !== 'io client disconnect') {
        reconnectAttempts.current += 1;
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 5000);
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
    newSocket.on('connect_error', (err) => {
      console.error('WebSocket connection error:', err);
      setError(err);
      onError?.(err);
    });

    // Gas price update
    newSocket.on('gas_price_update', (data: GasPriceUpdate) => {
      setGasPrice(data);
      setError(null);
    });

    // Connection established confirmation
    newSocket.on('connection_established', (data) => {
      console.log('Connection confirmed:', data);
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
  }, [enabled, onConnect, onDisconnect, onError]);

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
    error,
    disconnect,
    reconnect,
  };
}

