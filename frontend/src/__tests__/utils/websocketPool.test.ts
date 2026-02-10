import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { acquireSocket, releaseSocket, getSocketPoolStats, resetSocketPool } from '../../utils/websocketPool';

describe('websocketPool', () => {
  const socketA = { disconnect: vi.fn() };
  const socketB = { disconnect: vi.fn() };

  beforeEach(() => {
    resetSocketPool();
    socketA.disconnect.mockClear();
    socketB.disconnect.mockClear();
  });

  afterEach(() => {
    resetSocketPool();
  });

  it('reuses the same socket for the same key', () => {
    const factory = vi.fn(() => socketA as any);

    const first = acquireSocket('key-1', factory);
    const second = acquireSocket('key-1', factory);

    expect(first).toBe(second);
    expect(factory).toHaveBeenCalledTimes(1);
    expect(getSocketPoolStats('key-1').refCount).toBe(2);
  });

  it('disconnects only after the last release', () => {
    const factory = vi.fn(() => socketA as any);

    acquireSocket('key-2', factory);
    acquireSocket('key-2', factory);

    releaseSocket('key-2');
    expect(socketA.disconnect).not.toHaveBeenCalled();
    expect(getSocketPoolStats('key-2').refCount).toBe(1);

    releaseSocket('key-2');
    expect(socketA.disconnect).toHaveBeenCalledTimes(1);
    expect(getSocketPoolStats('key-2').refCount).toBe(0);
  });

  it('handles releases for unknown keys gracefully', () => {
    expect(() => releaseSocket('missing')).not.toThrow();
    expect(getSocketPoolStats('missing').size).toBe(0);
  });

  it('disconnects all sockets on reset', () => {
    acquireSocket('key-3', () => socketA as any);
    acquireSocket('key-4', () => socketB as any);

    resetSocketPool();

    expect(socketA.disconnect).toHaveBeenCalledTimes(1);
    expect(socketB.disconnect).toHaveBeenCalledTimes(1);
    expect(getSocketPoolStats('key-3').size).toBe(0);
  });
});
