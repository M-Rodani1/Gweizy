import { describe, it, expect, afterEach } from 'vitest';
import { registerDevTools, getDevTools } from '../../utils/devTools';

describe('development tools', () => {
  afterEach(() => {
    const win = window as typeof window & { __GWEIZY_DEVTOOLS__?: unknown };
    delete win.__GWEIZY_DEVTOOLS__;
  });

  it('registers devtools handle in dev mode', () => {
    const handle = registerDevTools('1.0.0');
    const stored = getDevTools();

    expect(handle).not.toBeNull();
    expect(stored?.version).toBe('1.0.0');
    expect(stored?.ping()).toBe('pong');
  });
});
