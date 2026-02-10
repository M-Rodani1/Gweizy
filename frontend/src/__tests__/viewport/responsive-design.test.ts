/**
 * Mobile Viewport and Responsive Design Tests
 *
 * Tests for responsive behavior, viewport-specific rendering,
 * and mobile-first design patterns.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

// ============================================================================
// Viewport Utilities
// ============================================================================

interface ViewportConfig {
  name: string;
  width: number;
  height: number;
  pixelRatio?: number;
  touch?: boolean;
}

// Common device viewports
const VIEWPORTS: Record<string, ViewportConfig> = {
  // Mobile devices
  iPhoneSE: { name: 'iPhone SE', width: 375, height: 667, pixelRatio: 2, touch: true },
  iPhone12: { name: 'iPhone 12', width: 390, height: 844, pixelRatio: 3, touch: true },
  iPhone14Pro: { name: 'iPhone 14 Pro', width: 393, height: 852, pixelRatio: 3, touch: true },
  pixel5: { name: 'Pixel 5', width: 393, height: 851, pixelRatio: 2.75, touch: true },
  galaxyS21: { name: 'Galaxy S21', width: 360, height: 800, pixelRatio: 3, touch: true },

  // Tablets
  iPadMini: { name: 'iPad Mini', width: 768, height: 1024, pixelRatio: 2, touch: true },
  iPadPro: { name: 'iPad Pro 11"', width: 834, height: 1194, pixelRatio: 2, touch: true },
  iPadPro12: { name: 'iPad Pro 12.9"', width: 1024, height: 1366, pixelRatio: 2, touch: true },

  // Desktop
  laptop: { name: 'Laptop', width: 1366, height: 768 },
  desktop: { name: 'Desktop', width: 1920, height: 1080 },
  desktop4K: { name: '4K Display', width: 3840, height: 2160 },
};

// Tailwind breakpoints
const TAILWIND_BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
};

/**
 * Check if viewport matches a media query
 */
function matchesMediaQuery(viewport: ViewportConfig, query: string): boolean {
  const minWidthMatch = query.match(/min-width:\s*(\d+)px/);
  const maxWidthMatch = query.match(/max-width:\s*(\d+)px/);
  const minHeightMatch = query.match(/min-height:\s*(\d+)px/);
  const maxHeightMatch = query.match(/max-height:\s*(\d+)px/);

  if (minWidthMatch && viewport.width < parseInt(minWidthMatch[1], 10)) return false;
  if (maxWidthMatch && viewport.width > parseInt(maxWidthMatch[1], 10)) return false;
  if (minHeightMatch && viewport.height < parseInt(minHeightMatch[1], 10)) return false;
  if (maxHeightMatch && viewport.height > parseInt(maxHeightMatch[1], 10)) return false;

  return true;
}

/**
 * Get active Tailwind breakpoint for viewport width
 */
function getActiveBreakpoint(width: number): string {
  if (width >= TAILWIND_BREAKPOINTS['2xl']) return '2xl';
  if (width >= TAILWIND_BREAKPOINTS.xl) return 'xl';
  if (width >= TAILWIND_BREAKPOINTS.lg) return 'lg';
  if (width >= TAILWIND_BREAKPOINTS.md) return 'md';
  if (width >= TAILWIND_BREAKPOINTS.sm) return 'sm';
  return 'base';
}

/**
 * Check if viewport is mobile-sized
 */
function isMobile(viewport: ViewportConfig): boolean {
  return viewport.width < TAILWIND_BREAKPOINTS.md;
}

/**
 * Check if viewport is tablet-sized
 */
function isTablet(viewport: ViewportConfig): boolean {
  return viewport.width >= TAILWIND_BREAKPOINTS.md && viewport.width < TAILWIND_BREAKPOINTS.lg;
}

/**
 * Check if viewport is desktop-sized
 */
function isDesktop(viewport: ViewportConfig): boolean {
  return viewport.width >= TAILWIND_BREAKPOINTS.lg;
}

/**
 * Check if viewport supports touch
 */
function isTouchDevice(viewport: ViewportConfig): boolean {
  return viewport.touch === true;
}

/**
 * Calculate responsive grid columns
 */
function getGridColumns(viewport: ViewportConfig): number {
  if (viewport.width < TAILWIND_BREAKPOINTS.sm) return 1;
  if (viewport.width < TAILWIND_BREAKPOINTS.md) return 2;
  if (viewport.width < TAILWIND_BREAKPOINTS.lg) return 3;
  if (viewport.width < TAILWIND_BREAKPOINTS.xl) return 4;
  return 6;
}

/**
 * Check if element would be visible based on responsive classes
 */
function isVisibleAtBreakpoint(classes: string, breakpoint: string): boolean {
  const classList = classes.split(' ');

  // Check for specific breakpoint hiding
  if (classList.includes(`${breakpoint}:hidden`)) {
    return false;
  }

  // For base breakpoint, check for plain 'hidden' without override
  if (breakpoint === 'base') {
    if (classList.includes('hidden')) {
      return false;
    }
    // Check for plain visibility classes
    return classList.includes('block') || classList.includes('flex') ||
           classList.includes('grid') || !classList.includes('hidden');
  }

  // For other breakpoints, check for hidden with override
  if (classList.includes('hidden')) {
    return classList.includes(`${breakpoint}:block`) ||
           classList.includes(`${breakpoint}:flex`) ||
           classList.includes(`${breakpoint}:grid`);
  }

  return true;
}

/**
 * Get font size scale for viewport
 */
function getFontScale(viewport: ViewportConfig): number {
  if (viewport.width < TAILWIND_BREAKPOINTS.sm) return 0.875; // 14px base
  if (viewport.width < TAILWIND_BREAKPOINTS.md) return 0.9375; // 15px base
  return 1; // 16px base
}

/**
 * Calculate safe areas for notched devices
 */
function getSafeAreas(viewport: ViewportConfig): { top: number; bottom: number; left: number; right: number } {
  // iPhone with notch/Dynamic Island
  if (viewport.name.includes('iPhone') && viewport.name !== 'iPhone SE') {
    return { top: 47, bottom: 34, left: 0, right: 0 };
  }

  // Standard devices
  return { top: 0, bottom: 0, left: 0, right: 0 };
}

// ============================================================================
// Tests
// ============================================================================

describe('Responsive Design Tests', () => {
  describe('Viewport Classification', () => {
    it('should classify mobile devices correctly', () => {
      expect(isMobile(VIEWPORTS.iPhoneSE)).toBe(true);
      expect(isMobile(VIEWPORTS.iPhone12)).toBe(true);
      expect(isMobile(VIEWPORTS.pixel5)).toBe(true);
      expect(isMobile(VIEWPORTS.galaxyS21)).toBe(true);
    });

    it('should classify tablets correctly', () => {
      expect(isTablet(VIEWPORTS.iPadMini)).toBe(true);
      expect(isTablet(VIEWPORTS.iPadPro)).toBe(true); // 834px is between md (768) and lg (1024)
      expect(isTablet(VIEWPORTS.iPadPro12)).toBe(false); // 1024px is lg
    });

    it('should classify desktops correctly', () => {
      expect(isDesktop(VIEWPORTS.laptop)).toBe(true);
      expect(isDesktop(VIEWPORTS.desktop)).toBe(true);
      expect(isDesktop(VIEWPORTS.desktop4K)).toBe(true);
    });

    it('should identify touch devices', () => {
      expect(isTouchDevice(VIEWPORTS.iPhoneSE)).toBe(true);
      expect(isTouchDevice(VIEWPORTS.iPadMini)).toBe(true);
      expect(isTouchDevice(VIEWPORTS.laptop)).toBe(false);
      expect(isTouchDevice(VIEWPORTS.desktop)).toBe(false);
    });
  });

  describe('Breakpoint Detection', () => {
    it('should detect base breakpoint for narrow screens', () => {
      expect(getActiveBreakpoint(320)).toBe('base');
      expect(getActiveBreakpoint(375)).toBe('base');
      expect(getActiveBreakpoint(639)).toBe('base');
    });

    it('should detect sm breakpoint', () => {
      expect(getActiveBreakpoint(640)).toBe('sm');
      expect(getActiveBreakpoint(700)).toBe('sm');
      expect(getActiveBreakpoint(767)).toBe('sm');
    });

    it('should detect md breakpoint', () => {
      expect(getActiveBreakpoint(768)).toBe('md');
      expect(getActiveBreakpoint(900)).toBe('md');
      expect(getActiveBreakpoint(1023)).toBe('md');
    });

    it('should detect lg breakpoint', () => {
      expect(getActiveBreakpoint(1024)).toBe('lg');
      expect(getActiveBreakpoint(1200)).toBe('lg');
      expect(getActiveBreakpoint(1279)).toBe('lg');
    });

    it('should detect xl breakpoint', () => {
      expect(getActiveBreakpoint(1280)).toBe('xl');
      expect(getActiveBreakpoint(1400)).toBe('xl');
      expect(getActiveBreakpoint(1535)).toBe('xl');
    });

    it('should detect 2xl breakpoint', () => {
      expect(getActiveBreakpoint(1536)).toBe('2xl');
      expect(getActiveBreakpoint(1920)).toBe('2xl');
      expect(getActiveBreakpoint(3840)).toBe('2xl');
    });
  });

  describe('Media Query Matching', () => {
    it('should match min-width queries', () => {
      expect(matchesMediaQuery(VIEWPORTS.iPhoneSE, '(min-width: 375px)')).toBe(true);
      expect(matchesMediaQuery(VIEWPORTS.iPhoneSE, '(min-width: 768px)')).toBe(false);
      expect(matchesMediaQuery(VIEWPORTS.desktop, '(min-width: 768px)')).toBe(true);
    });

    it('should match max-width queries', () => {
      expect(matchesMediaQuery(VIEWPORTS.iPhoneSE, '(max-width: 768px)')).toBe(true);
      expect(matchesMediaQuery(VIEWPORTS.desktop, '(max-width: 768px)')).toBe(false);
    });

    it('should match combined queries', () => {
      const tabletQuery = '(min-width: 768px) and (max-width: 1024px)';
      expect(matchesMediaQuery(VIEWPORTS.iPhoneSE, tabletQuery)).toBe(false);
      expect(matchesMediaQuery(VIEWPORTS.iPadMini, tabletQuery)).toBe(true);
      expect(matchesMediaQuery(VIEWPORTS.desktop, tabletQuery)).toBe(false);
    });

    it('should match height queries', () => {
      expect(matchesMediaQuery(VIEWPORTS.iPhoneSE, '(min-height: 600px)')).toBe(true);
      expect(matchesMediaQuery(VIEWPORTS.iPhoneSE, '(min-height: 800px)')).toBe(false);
    });
  });

  describe('Grid Layout Responsiveness', () => {
    it('should return 1 column for small screens', () => {
      expect(getGridColumns({ ...VIEWPORTS.iPhoneSE, width: 320 })).toBe(1);
      expect(getGridColumns({ ...VIEWPORTS.iPhoneSE, width: 375 })).toBe(1);
    });

    it('should return 2 columns for sm breakpoint', () => {
      expect(getGridColumns({ ...VIEWPORTS.iPhoneSE, width: 640 })).toBe(2);
      expect(getGridColumns({ ...VIEWPORTS.iPhoneSE, width: 700 })).toBe(2);
    });

    it('should return 3 columns for md breakpoint', () => {
      expect(getGridColumns(VIEWPORTS.iPadMini)).toBe(3);
    });

    it('should return 4 columns for lg breakpoint', () => {
      // Use a viewport in the lg range (1024-1279)
      expect(getGridColumns({ ...VIEWPORTS.laptop, width: 1100 })).toBe(4);
    });

    it('should return 6 columns for xl+ breakpoints', () => {
      expect(getGridColumns({ ...VIEWPORTS.desktop, width: 1280 })).toBe(6);
      expect(getGridColumns(VIEWPORTS.desktop)).toBe(6);
    });
  });

  describe('Responsive Visibility', () => {
    it('should handle hidden class with breakpoint override', () => {
      expect(isVisibleAtBreakpoint('hidden md:block', 'base')).toBe(false);
      expect(isVisibleAtBreakpoint('hidden md:block', 'md')).toBe(true);
    });

    it('should handle breakpoint-specific hiding', () => {
      expect(isVisibleAtBreakpoint('block md:hidden', 'base')).toBe(true);
      expect(isVisibleAtBreakpoint('block md:hidden', 'md')).toBe(false);
    });

    it('should handle flex/grid display', () => {
      expect(isVisibleAtBreakpoint('hidden lg:flex', 'lg')).toBe(true);
      expect(isVisibleAtBreakpoint('hidden lg:grid', 'lg')).toBe(true);
    });
  });

  describe('Font Scaling', () => {
    it('should return smaller scale for mobile', () => {
      expect(getFontScale(VIEWPORTS.iPhoneSE)).toBe(0.875);
      expect(getFontScale(VIEWPORTS.pixel5)).toBe(0.875);
    });

    it('should return medium scale for sm breakpoint', () => {
      expect(getFontScale({ ...VIEWPORTS.iPhoneSE, width: 650 })).toBe(0.9375);
    });

    it('should return full scale for desktop', () => {
      expect(getFontScale(VIEWPORTS.laptop)).toBe(1);
      expect(getFontScale(VIEWPORTS.desktop)).toBe(1);
    });
  });

  describe('Safe Areas', () => {
    it('should return safe areas for notched iPhones', () => {
      const areas = getSafeAreas(VIEWPORTS.iPhone12);
      expect(areas.top).toBeGreaterThan(0);
      expect(areas.bottom).toBeGreaterThan(0);
    });

    it('should return zero safe areas for iPhone SE', () => {
      const areas = getSafeAreas(VIEWPORTS.iPhoneSE);
      expect(areas.top).toBe(0);
      expect(areas.bottom).toBe(0);
    });

    it('should return zero safe areas for non-notched devices', () => {
      const areas = getSafeAreas(VIEWPORTS.pixel5);
      expect(areas.top).toBe(0);
      expect(areas.bottom).toBe(0);
    });
  });

  describe('Device-Specific Behavior', () => {
    it('should identify retina displays', () => {
      expect(VIEWPORTS.iPhone12.pixelRatio).toBe(3);
      expect(VIEWPORTS.pixel5.pixelRatio).toBe(2.75);
      expect(VIEWPORTS.laptop.pixelRatio).toBeUndefined();
    });

    it('should calculate actual pixel dimensions', () => {
      const logical = { width: VIEWPORTS.iPhone12.width, height: VIEWPORTS.iPhone12.height };
      const ratio = VIEWPORTS.iPhone12.pixelRatio!;
      const physical = { width: logical.width * ratio, height: logical.height * ratio };

      expect(physical.width).toBe(1170);
      expect(physical.height).toBe(2532);
    });
  });

  describe('Orientation Handling', () => {
    it('should detect portrait orientation', () => {
      const isPortrait = (v: ViewportConfig) => v.height > v.width;
      expect(isPortrait(VIEWPORTS.iPhoneSE)).toBe(true);
      expect(isPortrait(VIEWPORTS.iPhone12)).toBe(true);
    });

    it('should detect landscape orientation', () => {
      const isLandscape = (v: ViewportConfig) => v.width > v.height;
      const landscapeiPhone: ViewportConfig = {
        ...VIEWPORTS.iPhone12,
        width: VIEWPORTS.iPhone12.height,
        height: VIEWPORTS.iPhone12.width,
      };
      expect(isLandscape(landscapeiPhone)).toBe(true);
    });

    it('should handle orientation change', () => {
      const portrait = VIEWPORTS.iPadMini;
      const landscape: ViewportConfig = {
        ...portrait,
        width: portrait.height,
        height: portrait.width,
      };

      expect(getActiveBreakpoint(portrait.width)).toBe('md');
      expect(getActiveBreakpoint(landscape.width)).toBe('lg');
    });
  });

  describe('Aspect Ratio', () => {
    it('should calculate aspect ratio correctly', () => {
      const getAspectRatio = (v: ViewportConfig) => v.width / v.height;

      // iPhones have ~9:19.5 ratio
      expect(getAspectRatio(VIEWPORTS.iPhone12)).toBeCloseTo(0.46, 1);

      // iPads have ~3:4 ratio in portrait
      expect(getAspectRatio(VIEWPORTS.iPadMini)).toBeCloseTo(0.75, 1);

      // Desktops typically have 16:9 ratio
      expect(getAspectRatio(VIEWPORTS.desktop)).toBeCloseTo(1.78, 1);
    });
  });

  describe('Viewport Ranges', () => {
    it('should identify narrow viewports', () => {
      const isNarrow = (v: ViewportConfig) => v.width < 400;
      expect(isNarrow(VIEWPORTS.iPhoneSE)).toBe(true);
      expect(isNarrow(VIEWPORTS.galaxyS21)).toBe(true);
      expect(isNarrow(VIEWPORTS.iPhone12)).toBe(true);
    });

    it('should identify wide viewports', () => {
      const isWide = (v: ViewportConfig) => v.width >= 1920;
      expect(isWide(VIEWPORTS.desktop)).toBe(true);
      expect(isWide(VIEWPORTS.desktop4K)).toBe(true);
      expect(isWide(VIEWPORTS.laptop)).toBe(false);
    });

    it('should identify short viewports', () => {
      const isShort = (v: ViewportConfig) => v.height < 800;
      expect(isShort(VIEWPORTS.iPhoneSE)).toBe(true);
      expect(isShort(VIEWPORTS.laptop)).toBe(true);
      expect(isShort(VIEWPORTS.desktop)).toBe(false);
    });
  });

  describe('Touch Target Sizing', () => {
    it('should recommend minimum touch target size', () => {
      const getMinTouchTarget = (v: ViewportConfig) => {
        // Apple recommends 44pt, Google recommends 48dp
        return v.touch ? 44 : 24;
      };

      expect(getMinTouchTarget(VIEWPORTS.iPhone12)).toBe(44);
      expect(getMinTouchTarget(VIEWPORTS.laptop)).toBe(24);
    });

    it('should calculate touch target in pixels', () => {
      const getTouchTargetPixels = (v: ViewportConfig) => {
        const minSize = v.touch ? 44 : 24;
        return minSize * (v.pixelRatio || 1);
      };

      // iPhone 12 with 3x pixel ratio
      expect(getTouchTargetPixels(VIEWPORTS.iPhone12)).toBe(132);

      // Laptop without pixel ratio
      expect(getTouchTargetPixels(VIEWPORTS.laptop)).toBe(24);
    });
  });

  describe('Scroll Behavior', () => {
    it('should determine if horizontal scroll is needed', () => {
      const needsHorizontalScroll = (contentWidth: number, v: ViewportConfig) =>
        contentWidth > v.width;

      expect(needsHorizontalScroll(1200, VIEWPORTS.iPhoneSE)).toBe(true);
      expect(needsHorizontalScroll(1200, VIEWPORTS.desktop)).toBe(false);
    });

    it('should calculate visible content percentage', () => {
      const getVisiblePercentage = (contentWidth: number, v: ViewportConfig) =>
        Math.min(100, (v.width / contentWidth) * 100);

      // Mobile showing 400px content
      expect(getVisiblePercentage(400, VIEWPORTS.iPhoneSE)).toBeCloseTo(93.75, 1);

      // Desktop showing same content
      expect(getVisiblePercentage(400, VIEWPORTS.desktop)).toBe(100);
    });
  });

  describe('Container Width', () => {
    it('should calculate max container width based on breakpoint', () => {
      const getMaxContainerWidth = (v: ViewportConfig) => {
        const bp = getActiveBreakpoint(v.width);
        const maxWidths: Record<string, number> = {
          base: v.width,
          sm: 640,
          md: 768,
          lg: 1024,
          xl: 1280,
          '2xl': 1536,
        };
        return Math.min(v.width, maxWidths[bp]);
      };

      expect(getMaxContainerWidth(VIEWPORTS.iPhoneSE)).toBe(375);
      expect(getMaxContainerWidth(VIEWPORTS.laptop)).toBe(1280); // Laptop is 1366px (xl range)
      expect(getMaxContainerWidth(VIEWPORTS.desktop)).toBe(1536);
    });
  });
});
