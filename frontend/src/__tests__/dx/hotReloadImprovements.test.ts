import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('hot reload improvements', () => {
  it('enables HMR overlay in Vite config', () => {
    const configPath = resolve(__dirname, '../../../vite.config.ts');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain('hmr');
    expect(contents).toContain('overlay: true');
  });
});
