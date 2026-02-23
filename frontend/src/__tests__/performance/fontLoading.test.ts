import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

describe('Font loading optimization', () => {
  const htmlPath = path.resolve(process.cwd(), 'index.html');
  const htmlContent = fs.readFileSync(htmlPath, 'utf8');

  it('preconnects to Google Fonts origins', () => {
    expect(htmlContent).toContain('rel="preconnect" href="https://fonts.googleapis.com"');
    expect(htmlContent).toContain('rel="preconnect" href="https://fonts.gstatic.com"');
  });

  it('avoids remote font preloads that can produce unused preload warnings', () => {
    expect(htmlContent).not.toMatch(/rel="preload"[^>]+as="font"/i);
  });

  it('requests fonts with font-display swap', () => {
    expect(htmlContent).toMatch(/fonts\.googleapis\.com\/css2\?[^"']*display=swap/);
  });
});
