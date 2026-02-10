import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('naming convention enforcement', () => {
  it('configures @typescript-eslint/naming-convention', () => {
    const configPath = resolve(__dirname, '../../../eslint.config.js');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain("'@typescript-eslint/naming-convention'");
    expect(contents).toContain("selector: 'typeLike'");
    expect(contents).toContain("format: ['PascalCase']");
  });
});
