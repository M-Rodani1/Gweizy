import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('api documentation generation', () => {
  it('includes api docs generation script', () => {
    const scriptPath = resolve(__dirname, '../../../scripts/generateApiDocs.js');
    const contents = readFileSync(scriptPath, 'utf8');

    expect(contents).toContain('API docs generated');
    expect(contents).toContain('API Reference');
  });

  it('registers docs:api npm script', () => {
    const pkgPath = resolve(__dirname, '../../../package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { scripts?: Record<string, string> };

    expect(pkg.scripts?.['docs:api']).toBe('node scripts/generateApiDocs.js');
  });
});
