import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('e2e test framework', () => {
  it('includes Playwright configuration', () => {
    const configPath = resolve(__dirname, '../../../playwright.config.ts');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain('playwright');
    expect(contents).toContain('defineConfig');
  });
});
