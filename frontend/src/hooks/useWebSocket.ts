/**
 * WebSocket hook for real-time gas price updates
 * Replaces polling with WebSocket connections for instant updates
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { getApiOrigin } from '../config/api';

interface GasPriceUpdate {
  current_gas: number;
  base_fee: number;
  priority_fee: number;
  timestamp: string;
  collection_count: number;
}

interface UseWebSocketOptions {
  enabled?: boolean;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
}

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
    const socketUrl = apiOrigin.replace(/^https?:\/\//, 'ws://').replace(/^https:\/\//, 'wss://');
    
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

