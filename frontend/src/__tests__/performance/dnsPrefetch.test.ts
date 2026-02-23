import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

describe('DNS prefetch hints', () => {
  const htmlPath = path.resolve(process.cwd(), 'index.html');
  const htmlContent = fs.readFileSync(htmlPath, 'utf8');

  it('prefetches DNS for external RPC domains', () => {
    expect(htmlContent).toContain('rel="dns-prefetch" href="https://mainnet.base.org"');
    expect(htmlContent).toContain('rel="dns-prefetch" href="https://base-rpc.publicnode.com"');
  });
});
