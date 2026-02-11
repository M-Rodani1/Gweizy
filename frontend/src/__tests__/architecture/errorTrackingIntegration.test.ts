import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('error tracking integration', () => {
  it('initializes Sentry in main entrypoint', () => {
    const entryPath = resolve(__dirname, '../../../src/main.tsx');
    const contents = readFileSync(entryPath, 'utf8');

    expect(contents).toContain("import('@sentry/react')");
    expect(contents).toContain('Sentry.init');
  });
});
