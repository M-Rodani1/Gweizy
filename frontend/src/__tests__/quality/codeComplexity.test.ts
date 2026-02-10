import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('code complexity metrics', () => {
  it('configures ESLint complexity rule', () => {
    const configPath = resolve(__dirname, '../../../eslint.config.js');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain("'complexity'");
    expect(contents).toContain('max: 12');
  });
});
