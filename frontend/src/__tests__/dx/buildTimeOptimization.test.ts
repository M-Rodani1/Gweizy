import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('build time optimization', () => {
  it('disables compressed size reporting for faster builds', () => {
    const configPath = resolve(__dirname, '../../../vite.config.ts');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain('reportCompressedSize: false');
  });
});
