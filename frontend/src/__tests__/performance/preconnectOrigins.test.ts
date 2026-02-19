import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

describe('Preconnect origins', () => {
  const htmlPath = path.resolve(process.cwd(), 'index.html');
  const htmlContent = fs.readFileSync(htmlPath, 'utf8');

  it('preconnects to frequently used RPC origins', () => {
    expect(htmlContent).toContain('rel="preconnect" href="https://mainnet.base.org"');
    expect(htmlContent).toContain('rel="preconnect" href="https://base.llamarpc.com"');
    expect(htmlContent).toContain('rel="preconnect" href="https://base-rpc.publicnode.com"');
  });
});
