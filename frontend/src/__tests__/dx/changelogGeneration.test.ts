import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('changelog generation', () => {
  it('registers changelog script', () => {
    const pkgPath = resolve(__dirname, '../../../package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { scripts?: Record<string, string> };

    expect(pkg.scripts?.changelog).toBe('node scripts/generateChangelog.js');
  });

  it('includes changelog generator script', () => {
    const scriptPath = resolve(__dirname, '../../../scripts/generateChangelog.js');
    const contents = readFileSync(scriptPath, 'utf8');

    expect(contents).toContain('Changelog updated');
  });
});
