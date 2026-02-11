import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

describe('Dynamic imports for heavy libraries', () => {
  it('lazy-loads chart rendering for accuracy dashboard', () => {
    const filePath = path.resolve(process.cwd(), 'src/components/AccuracyDashboard.tsx');
    const contents = fs.readFileSync(filePath, 'utf8');

    expect(contents).toMatch(/lazy\(\(\) => import\('\.\/AccuracyDashboardCharts'\)\)/);
  });
});
