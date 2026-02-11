import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('storybook integration', () => {
  it('configures Storybook for Vite', () => {
    const configPath = resolve(__dirname, '../../../.storybook/main.ts');
    const contents = readFileSync(configPath, 'utf8');

    expect(contents).toContain('@storybook/react-vite');
  });
});
