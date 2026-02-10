import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('TypeScript strict configuration', () => {
  it('enables strict mode in tsconfig', () => {
    const tsconfigPath = resolve(__dirname, '../../../tsconfig.json');
    const contents = readFileSync(tsconfigPath, 'utf8');

    expect(contents).toContain('"strict": true');
    expect(contents).toContain('"strictNullChecks": true');
    expect(contents).toContain('"noImplicitAny": true');
  });

  it('enables additional strict flags in tsconfig.strict.json', () => {
    const strictPath = resolve(__dirname, '../../../tsconfig.strict.json');
    const contents = readFileSync(strictPath, 'utf8');

    expect(contents).toContain('"noUncheckedIndexedAccess": true');
    expect(contents).toContain('"exactOptionalPropertyTypes": true');
    expect(contents).toContain('"noPropertyAccessFromIndexSignature": true');
  });
});
