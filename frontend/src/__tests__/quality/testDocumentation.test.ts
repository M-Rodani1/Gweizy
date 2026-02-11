import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('test documentation generation', () => {
  it('includes test docs generation script', () => {
    const scriptPath = resolve(__dirname, '../../../scripts/generateTestDocs.js');
    const contents = readFileSync(scriptPath, 'utf8');

    expect(contents).toContain('Test docs generated');
    expect(contents).toContain('Test Reference');
  });

  it('registers docs:tests npm script', () => {
    const pkgPath = resolve(__dirname, '../../../package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { scripts?: Record<string, string> };

    expect(pkg.scripts?.['docs:tests']).toBe('node scripts/generateTestDocs.js');
  });
});
