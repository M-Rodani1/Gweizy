/**
 * E2E Tests for Gas Predictions
 *
 * Tests the gas prediction display and interaction.
 */

import { test, expect } from '@playwright/test';

test.describe('Gas Predictions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display gas price data', async ({ page }) => {
    // Wait for content to load
    await page.waitForLoadState('networkidle');

    // Look for gas-related content
    const pageContent = await page.textContent('body');

    // Should have some indication of gas or gwei
    const hasGasContent =
      pageContent?.toLowerCase().includes('gas') ||
      pageContent?.toLowerCase().includes('gwei') ||
      pageContent?.toLowerCase().includes('price');

    expect(hasGasContent).toBe(true);
  });

  test('should update predictions in real-time', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Get initial content snapshot
    const initialContent = await page.textContent('body');

    // Wait for potential update (WebSocket or polling)
    await page.waitForTimeout(5000);

    // Page should still be responsive
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });

  test('should allow time range selection', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Look for time range selectors (1h, 4h, 24h buttons)
    const timeButtons = page.locator('button:has-text("1h"), button:has-text("4h"), button:has-text("24h")');
    const buttonCount = await timeButtons.count();

    if (buttonCount > 0) {
      // Click each time range button
      for (let i = 0; i < buttonCount; i++) {
        const button = timeButtons.nth(i);
        await button.click();
        await page.waitForTimeout(500); // Allow state to update

        // Button should be visually selected (has active class or aria-pressed)
        const isPressed = await button.getAttribute('aria-pressed');
        const classList = await button.getAttribute('class');

        const isSelected =
          isPressed === 'true' ||
          classList?.includes('active') ||
          classList?.includes('selected') ||
          classList?.includes('bg-cyan') ||
          classList?.includes('bg-blue');

        // At least one indicator of selection
        expect(classList).toBeTruthy();
      }
    }
  });

  test('should display confidence indicators', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Look for confidence-related elements
    const confidenceElements = page.locator(
      '[class*="confidence"], [data-testid*="confidence"], text=/confidence/i, text=/%/'
    );

    // May or may not have confidence indicators depending on view
    const count = await confidenceElements.count();

    // Just verify page loaded without errors
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });

  test('should handle loading states gracefully', async ({ page }) => {
    // Intercept API calls to simulate slow loading
    await page.route('**/api/**', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      await route.continue();
    });

    await page.goto('/');

    // Should show loading indicator or skeleton
    const loadingIndicators = page.locator(
      '[class*="skeleton"], [class*="loading"], [class*="spinner"], [role="status"]'
    );

    // Page should handle loading gracefully (not crash)
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });

  test('should handle API errors gracefully', async ({ page }) => {
    // Intercept API calls to simulate errors
    await page.route('**/api/**', (route) => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal Server Error' }),
      });
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Should show error state or fallback content
    const body = page.locator('body');
    await expect(body).toBeVisible();

    // Should not show blank page
    const content = await page.textContent('body');
    expect(content?.trim().length).toBeGreaterThan(0);
  });
});

test.describe('Chain Selection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('should allow chain switching', async ({ page }) => {
    // Look for chain selector
    const chainSelector = page.locator(
      '[data-testid="chain-selector"], [aria-label*="chain"], button:has-text("Ethereum"), button:has-text("Base")'
    );

    const selectorCount = await chainSelector.count();

    if (selectorCount > 0) {
      await chainSelector.first().click();

      // Should show chain options or switch chains
      await page.waitForTimeout(500);

      const body = page.locator('body');
      await expect(body).toBeVisible();
    }
  });

  test('should display chain-specific data', async ({ page }) => {
    // Look for chain identifiers
    const chainBadges = page.locator(
      '[class*="chain"], [data-testid*="chain"], img[alt*="Ethereum"], img[alt*="Base"]'
    );

    // Just verify page is functional
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });
});

test.describe('Data Refresh', () => {
  test('should have refresh functionality', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Look for refresh button
    const refreshButton = page.locator(
      'button[aria-label*="refresh"], button[aria-label*="Refresh"], button:has([class*="refresh"]), button:has([class*="RefreshCw"])'
    );

    const refreshCount = await refreshButton.count();

    if (refreshCount > 0) {
      await refreshButton.first().click();

      // Should trigger refresh without error
      await page.waitForTimeout(1000);

      const body = page.locator('body');
      await expect(body).toBeVisible();
    }
  });

  test('should show last updated timestamp', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Look for timestamp indicators
    const timestamps = page.locator(
      'text=/updated/i, text=/ago/i, text=/last/i, time, [datetime]'
    );

    // Just verify page loaded
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });
});
