import { describe, it, expect, vi, beforeEach } from 'vitest';
import { enqueueOfflineAction, flushOfflineQueue, getOfflineQueue } from '../../utils/offlineQueue';

describe('offline-first architecture', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: vi.fn(),
      setItem: vi.fn(),
      removeItem: vi.fn()
    });
  });

  it('queues actions when offline', () => {
    enqueueOfflineAction({ type: 'sync', payload: { id: 1 } });
    expect(localStorage.setItem).toHaveBeenCalled();
  });

  it('flushes queued actions', async () => {
    (localStorage.getItem as ReturnType<typeof vi.fn>).mockReturnValue('[{"type":"sync","payload":{"id":1}}]');
    const handler = vi.fn().mockResolvedValue(undefined);

    await flushOfflineQueue(handler);

    expect(handler).toHaveBeenCalledWith({ type: 'sync', payload: { id: 1 } });
    expect(getOfflineQueue()).toEqual([{ type: 'sync', payload: { id: 1 } }]);
  });
});
