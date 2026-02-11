import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('circular dependency detection', () => {
  it('includes circular dependency check script', () => {
    const scriptPath = resolve(__dirname, '../../../scripts/checkCircularDeps.js');
    const contents = readFileSync(scriptPath, 'utf8');

    expect(contents).toContain('Circular dependencies detected');
    expect(contents).toContain('No circular dependencies detected');
  });

  it('registers lint:circular npm script', () => {
    const pkgPath = resolve(__dirname, '../../../package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { scripts?: Record<string, string> };

    expect(pkg.scripts?.['lint:circular']).toBe('node scripts/checkCircularDeps.js');
  });
});
