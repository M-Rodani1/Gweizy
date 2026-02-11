import { describe, it, expect, vi } from 'vitest';
import { PluginRegistry } from '../../utils/pluginRegistry';

describe('plugin architecture', () => {
  it('registers and initializes plugins', () => {
    const registry = new PluginRegistry();
    const setup = vi.fn();

    registry.register({ name: 'test', setup });
    registry.initializeAll();

    expect(setup).toHaveBeenCalled();
    expect(registry.list()).toHaveLength(1);
  });
});
