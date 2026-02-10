import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { prefersReducedMotion } from '../../utils/performanceOptimizations';

describe('reduced motion support', () => {
  it('exposes prefersReducedMotion utility', () => {
    expect(typeof prefersReducedMotion).toBe('function');
  });

  it('defines reduced motion media query in CSS', () => {
    const cssPath = path.resolve(process.cwd(), 'src/index.css');
    const css = fs.readFileSync(cssPath, 'utf8');
    expect(css).toContain('@media (prefers-reduced-motion: reduce)');
    expect(css).toContain('@media (prefers-contrast: high)');
  });
});
