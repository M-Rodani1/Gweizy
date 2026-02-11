import { describe, it, expect, vi } from 'vitest';
import { loadRemoteModule } from '../../utils/moduleFederation';

describe('module federation support', () => {
  it('loads module from remote container', async () => {
    const init = vi.fn();
    const get = vi.fn().mockResolvedValue(() => ({ value: 42 }));

    vi.stubGlobal('window', {
      remoteApp: { init, get }
    });

    const module = await loadRemoteModule<{ value: number }>('remoteApp', './Widget');
    expect(init).toHaveBeenCalled();
    expect(get).toHaveBeenCalledWith('./Widget');
    expect(module.value).toBe(42);
  });

  it('throws when remote container is missing', async () => {
    vi.stubGlobal('window', {});
    await expect(loadRemoteModule('missing', './Widget')).rejects.toThrow('Remote container not found');
  });
});
