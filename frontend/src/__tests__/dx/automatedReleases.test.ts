import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('automated releases', () => {
  it('registers release preparation script', () => {
    const pkgPath = resolve(__dirname, '../../../package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { scripts?: Record<string, string> };

    expect(pkg.scripts?.['release:prepare']).toBe('node scripts/prepareRelease.js');
  });

  it('includes release notes generator script', () => {
    const scriptPath = resolve(__dirname, '../../../scripts/prepareRelease.js');
    const contents = readFileSync(scriptPath, 'utf8');

    expect(contents).toContain('Release notes prepared');
  });
});
