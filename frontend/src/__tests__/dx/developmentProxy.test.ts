import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('development proxy setup', () => {
  it('configures /api proxy in Vite', () => {
    const configPath = resolve(__dirname, '../../../vite.config.ts');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain("'/api'");
    expect(contents).toContain('changeOrigin: true');
  });
});
