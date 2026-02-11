import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('ci/cd pipeline improvements', () => {
  it('adds concurrency controls to CI workflow', () => {
    const workflowPath = resolve(__dirname, '../../../../.github/workflows/ci.yml');
    const contents = readFileSync(workflowPath, 'utf8');

    expect(contents).toContain('concurrency:');
    expect(contents).toContain('cancel-in-progress');
  });
});
