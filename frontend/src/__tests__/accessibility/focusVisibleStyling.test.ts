import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

describe('focus visible styling', () => {
  it('defines focus-visible styles in CSS', () => {
    const cssPath = path.resolve(process.cwd(), 'src/index.css');
    const css = fs.readFileSync(cssPath, 'utf8');
    expect(css).toContain('*:focus-visible');
    expect(css).toContain('button:focus-visible');
  });
});
