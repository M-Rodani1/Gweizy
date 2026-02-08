/**
 * Storybook test-runner configuration for visual regression testing
 */

import type { TestRunnerConfig } from '@storybook/test-runner';
import { toMatchImageSnapshot } from 'jest-image-snapshot';

const config: TestRunnerConfig = {
  setup() {
    // Extend expect with image snapshot matcher
    expect.extend({ toMatchImageSnapshot });
  },
  async postVisit(page, context) {
    // Wait for animations to complete
    await page.waitForTimeout(100);

    // Capture screenshot for visual regression
    const image = await page.screenshot();
    expect(image).toMatchImageSnapshot({
      customSnapshotsDir: `${process.cwd()}/__snapshots__`,
      customSnapshotIdentifier: context.id,
      failureThreshold: 0.01,
      failureThresholdType: 'percent',
    });
  },
};

export default config;
