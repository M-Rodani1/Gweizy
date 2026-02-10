import { describe, it, expect } from 'vitest';
import { evaluateBundleBudget } from '../../../scripts/checkBundleBudget.js';

describe('bundle budget evaluation', () => {
  it('passes when bundle is within budget', () => {
    const report = {
      totalBytes: 1000,
      topFiles: [{ filePath: 'app.js', bytes: 600 }],
    };
    const budget = { totalMaxBytes: 2000, largestAssetMaxBytes: 1000 };

    const result = evaluateBundleBudget(report, budget);

    expect(result.ok).toBe(true);
    expect(result.failures).toHaveLength(0);
  });

  it('fails when bundle exceeds budget', () => {
    const report = {
      totalBytes: 5000,
      topFiles: [{ filePath: 'app.js', bytes: 3000 }],
    };
    const budget = { totalMaxBytes: 2000, largestAssetMaxBytes: 1000 };

    const result = evaluateBundleBudget(report, budget);

    expect(result.ok).toBe(false);
    expect(result.failures.length).toBeGreaterThan(0);
  });
});
