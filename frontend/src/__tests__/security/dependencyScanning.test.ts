import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

describe('dependency vulnerability scanning', () => {
  it('defines audit script in package.json', () => {
    const pkgPath = path.resolve(process.cwd(), 'package.json');
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8')) as {
      scripts?: Record<string, string>;
    };

    expect(pkg.scripts?.['audit:dependencies']).toContain('npm audit');
  });
});
