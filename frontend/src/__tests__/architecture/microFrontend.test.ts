import { describe, it, expect, vi } from 'vitest';
import { registerMicroFrontend, mountMicroFrontend, listMicroFrontends } from '../../utils/microFrontend';

describe('micro-frontend preparation', () => {
  it('registers and mounts micro frontends', () => {
    const mount = vi.fn();
    registerMicroFrontend({ name: 'shell', mount });

    mountMicroFrontend('shell', document.body);

    expect(mount).toHaveBeenCalledWith(document.body);
    expect(listMicroFrontends().length).toBeGreaterThan(0);
  });
});
