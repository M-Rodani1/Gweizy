import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('component documentation generation', () => {
  it('includes component docs generation script', () => {
    const scriptPath = resolve(__dirname, '../../../scripts/generateComponentDocs.js');
    const contents = readFileSync(scriptPath, 'utf8');

    expect(contents).toContain('Component docs generated');
    expect(contents).toContain('Component Reference');
  });

  it('registers docs:components npm script', () => {
    const pkgPath = resolve(__dirname, '../../../package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { scripts?: Record<string, string> };

    expect(pkg.scripts?.['docs:components']).toBe('node scripts/generateComponentDocs.js');
  });
});
