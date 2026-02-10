import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createBundleSizeReport, formatBytes } from '../../../scripts/bundleSizeReport.js';

describe('bundle size monitoring', () => {
  it('creates a size report with totals and top files', () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'bundle-size-'));
    const files = [
      { name: 'a.js', size: 10 },
      { name: 'b.js', size: 50 },
      { name: 'c.css', size: 5 },
      { name: 'nested/d.js', size: 20 },
    ];

    for (const file of files) {
      const target = path.join(tempDir, file.name);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, 'x'.repeat(file.size));
    }

    const report = createBundleSizeReport(tempDir, { topN: 2 });

    expect(report.totalBytes).toBe(85);
    expect(report.fileCount).toBe(4);
    expect(report.topFiles).toHaveLength(2);
    expect(report.topFiles[0].bytes).toBe(50);
    expect(report.topFiles[1].bytes).toBe(20);
  });

  it('formats byte sizes for display', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(2048)).toBe('2.00 KB');
  });
});
