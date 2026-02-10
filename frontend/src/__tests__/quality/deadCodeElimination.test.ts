import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('dead code elimination configuration', () => {
  it('marks known side-effect files to enable tree-shaking', () => {
    const pkgPath = resolve(__dirname, '../../../package.json');
    const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as { sideEffects?: string[] };

    expect(pkg.sideEffects).toEqual(
      expect.arrayContaining([
        '**/*.css',
        'lucide-react/dist/esm/**/*.js',
        'lucide-react/**/*.js'
      ])
    );
  });
});
