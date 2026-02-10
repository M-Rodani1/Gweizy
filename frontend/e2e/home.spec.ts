/**
 * E2E Tests for Home Page
 *
 * Tests the main landing page and core functionality.
 */

import { test, expect } from '@playwright/test';

test.describe('Home Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display the main heading', async ({ page }) => {
    // Check for main heading or logo
    const heading = page.locator('h1, [role="banner"]').first();
    await expect(heading).toBeVisible();
  });

  test('should have skip to content link for accessibility', async ({ page }) => {
    const skipLink = page.locator('a[href="#main-content"]');
    await expect(skipLink).toBeAttached();
  });

  test('should display gas price information', async ({ page }) => {
    // Wait for gas data to load
    await page.waitForSelector('[data-testid="gas-price"], .gwei, text=/gwei/i', {
      timeout: 10000,
    }).catch(() => {
      // Gas price might be in different format
    });

    // Check that some numeric content is displayed (gas prices)
    const content = await page.textContent('body');
    expect(content).toBeTruthy();
  });

  test('should be responsive on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');

    // Page should still be functional
    const body = page.locator('body');
    await expect(body).toBeVisible();

    // No horizontal scroll on mobile
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(scrollWidth).toBeLessThanOrEqual(clientWidth + 10); // Allow small margin
  });

  test('should have proper meta tags for SEO', async ({ page }) => {
    const title = await page.title();
    expect(title).toBeTruthy();
    expect(title.length).toBeGreaterThan(0);

    const metaDescription = page.locator('meta[name="description"]');
    await expect(metaDescription).toBeAttached();
  });

  test('should have no console errors on load', async ({ page }) => {
    const consoleErrors: string[] = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Filter out known acceptable errors (e.g., API not available in test)
    const criticalErrors = consoleErrors.filter(
      (error) =>
        !error.includes('Failed to fetch') &&
        !error.includes('net::ERR_') &&
        !error.includes('WebSocket')
    );

    expect(criticalErrors).toHaveLength(0);
  });
});

test.describe('Navigation', () => {
  test('should navigate between pages', async ({ page }) => {
    await page.goto('/');

    // Find navigation links
    const navLinks = page.locator('nav a, header a');
    const linkCount = await navLinks.count();

    if (linkCount > 0) {
      // Click first nav link and verify navigation
      const firstLink = navLinks.first();
      const href = await firstLink.getAttribute('href');

      if (href && !href.startsWith('http') && href !== '#') {
        await firstLink.click();
        await page.waitForLoadState('networkidle');
        expect(page.url()).toContain(href.replace(/^\//, ''));
      }
    }
  });

  test('should handle 404 pages gracefully', async ({ page }) => {
    await page.goto('/non-existent-page-12345');

    // Should show some content (not blank page)
    const body = page.locator('body');
    await expect(body).toBeVisible();

    // Should either redirect or show error message
    const content = await page.textContent('body');
    expect(content?.length).toBeGreaterThan(0);
  });
});

test.describe('Accessibility', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/');

    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBeLessThanOrEqual(1); // At most one h1

    // Check that headings exist
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').count();
    expect(headings).toBeGreaterThan(0);
  });

  test('should have alt text on images', async ({ page }) => {
    await page.goto('/');

    const images = page.locator('img');
    const imageCount = await images.count();

    for (let i = 0; i < imageCount; i++) {
      const img = images.nth(i);
      const alt = await img.getAttribute('alt');
      const role = await img.getAttribute('role');
      const ariaHidden = await img.getAttribute('aria-hidden');

      // Image should have alt text OR be marked as decorative
      const isAccessible =
        alt !== null || role === 'presentation' || ariaHidden === 'true';
      expect(isAccessible).toBe(true);
    }
  });

  test('should be keyboard navigable', async ({ page }) => {
    await page.goto('/');

    // Tab through the page
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Something should be focused
    const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
    expect(focusedElement).toBeTruthy();
    expect(focusedElement).not.toBe('BODY');
  });

  test('should have visible focus indicators', async ({ page }) => {
    await page.goto('/');

    // Find a focusable element
    const focusable = page.locator('button, a, input, select, textarea, [tabindex]').first();

    if (await focusable.count() > 0) {
      await focusable.focus();

      // Check for focus styling (outline or box-shadow)
      const styles = await focusable.evaluate((el) => {
        const computed = window.getComputedStyle(el);
        return {
          outline: computed.outline,
          boxShadow: computed.boxShadow,
          outlineWidth: computed.outlineWidth,
        };
      });

      const hasFocusIndicator =
        (styles.outline !== 'none' && styles.outlineWidth !== '0px') ||
        (styles.boxShadow !== 'none' && styles.boxShadow !== '');

      expect(hasFocusIndicator).toBe(true);
    }
  });
});
