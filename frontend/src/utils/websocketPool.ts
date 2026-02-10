import type { Socket } from 'socket.io-client';

type SocketFactory = () => Socket;

interface PoolEntry {
  socket: Socket;
  refCount: number;
}

const pool = new Map<string, PoolEntry>();

export function acquireSocket(key: string, factory: SocketFactory): Socket {
  const existing = pool.get(key);
  if (existing) {
    existing.refCount += 1;
    return existing.socket;
  }

  const socket = factory();
  pool.set(key, { socket, refCount: 1 });
  return socket;
}

export function releaseSocket(key: string): void {
  const existing = pool.get(key);
  if (!existing) {
    return;
  }

  existing.refCount -= 1;
  if (existing.refCount <= 0) {
    existing.socket.disconnect();
    pool.delete(key);
  }
}

export function getSocketPoolStats(key: string): { size: number; refCount: number } {
  const entry = pool.get(key);
  return { size: pool.size, refCount: entry ? entry.refCount : 0 };
}

export function resetSocketPool(): void {
  for (const entry of pool.values()) {
    entry.socket.disconnect();
  }
  pool.clear();
}
