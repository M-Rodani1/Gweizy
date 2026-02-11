import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

describe('Compression verification', () => {
  it('enables compressed size reporting for Brotli verification', () => {
    const configPath = path.resolve(process.cwd(), 'vite.config.ts');
    const contents = fs.readFileSync(configPath, 'utf8');

    expect(contents).toMatch(/reportCompressedSize:\s*true/);
  });
});
