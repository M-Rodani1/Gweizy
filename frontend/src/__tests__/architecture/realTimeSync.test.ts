import { describe, it, expect, vi } from 'vitest';
import { RealTimeSync } from '../../utils/realTimeSync';

describe('real-time sync architecture', () => {
  it('notifies subscribers on publish', () => {
    const sync = new RealTimeSync<string>();
    const callback = vi.fn();

    sync.subscribe(callback);
    sync.publish('update');

    expect(callback).toHaveBeenCalledWith('update');
  });

  it('allows unsubscribe', () => {
    const sync = new RealTimeSync<number>();
    const callback = vi.fn();

    const unsubscribe = sync.subscribe(callback);
    unsubscribe();
    sync.publish(1);

    expect(callback).not.toHaveBeenCalled();
  });
});
