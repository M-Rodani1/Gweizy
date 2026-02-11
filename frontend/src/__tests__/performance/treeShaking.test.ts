import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

describe('Tree-shaking verification', () => {
  it('enables Rollup treeshake configuration', () => {
    const configPath = path.resolve(process.cwd(), 'vite.config.ts');
    const contents = fs.readFileSync(configPath, 'utf8');

    expect(contents).toMatch(/treeshake:\s*true/);
  });
});
