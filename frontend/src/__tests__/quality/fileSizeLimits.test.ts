import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('file size limits', () => {
  it('configures max-lines rule in ESLint', () => {
    const configPath = resolve(__dirname, '../../../eslint.config.js');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain("'max-lines'");
    expect(contents).toContain('max: 500');
  });
});
