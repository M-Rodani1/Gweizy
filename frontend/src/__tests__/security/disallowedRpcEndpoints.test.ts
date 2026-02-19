import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const DISALLOWED_RPC_HOSTS = [
  'polygon-rpc.com',
  'polygon.llamarpc.com',
  'polygon-bor.publicnode.com'
];

describe('disallowed RPC endpoint guardrails', () => {
  it('does not include known failing Polygon RPC hosts in source config', () => {
    const files = [
      path.resolve(process.cwd(), 'src/config/chains.ts'),
      path.resolve(process.cwd(), 'src/config/api.ts'),
      path.resolve(process.cwd(), 'public/_headers'),
      path.resolve(process.cwd(), 'index.html')
    ];

    for (const filePath of files) {
      const contents = fs.readFileSync(filePath, 'utf8');
      for (const host of DISALLOWED_RPC_HOSTS) {
        expect(contents).not.toContain(host);
      }
    }
  });
});
