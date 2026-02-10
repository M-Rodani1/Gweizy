import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('import organization linting', () => {
  it('configures sort-imports rule for member ordering', () => {
    const configPath = resolve(__dirname, '../../../eslint.config.js');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain("'sort-imports'");
    expect(contents).toContain('ignoreDeclarationSort');
    expect(contents).toContain('ignoreMemberSort');
  });
});
