import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('visual regression testing', () => {
  it('exposes chromatic scripts', () => {
    const pkgPath = resolve(__dirname, '../../../package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { scripts?: Record<string, string> };

    expect(pkg.scripts?.chromatic).toBeDefined();
    expect(pkg.scripts?.['chromatic:ci']).toBeDefined();
  });
});
