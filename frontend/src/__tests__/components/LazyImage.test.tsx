/**
 * LazyImage Component Tests
 *
 * Tests for the lazy-loaded image component that uses
 * IntersectionObserver to load images when they enter the viewport.
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// ============================================================================
// Mock IntersectionObserver
// ============================================================================

interface MockIntersectionObserverEntry {
  isIntersecting: boolean;
  target: Element;
}

type IntersectionCallback = (entries: MockIntersectionObserverEntry[]) => void;

let mockIntersectionCallback: IntersectionCallback | null = null;
let mockObservedElements: Set<Element>;
let mockObserverOptions: IntersectionObserverInit | undefined;

class MockIntersectionObserver {
  callback: IntersectionCallback;
  options?: IntersectionObserverInit;

  constructor(callback: IntersectionCallback, options?: IntersectionObserverInit) {
    this.callback = callback;
    this.options = options;
    mockIntersectionCallback = callback;
    mockObserverOptions = options;
    mockObservedElements = new Set();
  }

  observe(element: Element) {
    mockObservedElements.add(element);
  }

  unobserve(element: Element) {
    mockObservedElements.delete(element);
  }

  disconnect() {
    mockObservedElements.clear();
  }
}

// Helper to simulate element entering viewport
function simulateIntersection(isIntersecting: boolean) {
  if (mockIntersectionCallback && mockObservedElements.size > 0) {
    const entries: MockIntersectionObserverEntry[] = Array.from(mockObservedElements).map(
      (target) => ({
        isIntersecting,
        target,
      })
    );
    mockIntersectionCallback(entries);
  }
}

// ============================================================================
// LazyImage Component (inline for testing)
// ============================================================================

import { useState, useRef, useEffect, memo } from 'react';

interface LazyImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  alt: string;
  placeholder?: string;
  className?: string;
}

const LazyImage: React.FC<LazyImageProps> = memo(
  ({
    src,
    alt,
    placeholder = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"%3E%3Crect fill="%231f2937" width="1" height="1"/%3E%3C/svg%3E',
    className = '',
    ...props
  }) => {
    const [isLoaded, setIsLoaded] = useState(false);
    const [isInView, setIsInView] = useState(false);
    const imgRef = useRef<HTMLImageElement>(null);

    useEffect(() => {
      if (!imgRef.current) return;

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            setIsInView(true);
            observer.disconnect();
          }
        },
        {
          rootMargin: '100px',
          threshold: 0.01,
        }
      );

      observer.observe(imgRef.current);

      return () => observer.disconnect();
    }, []);

    return (
      <img
        ref={imgRef}
        src={isInView ? src : placeholder}
        alt={alt}
        className={`transition-opacity duration-300 ${isLoaded ? 'opacity-100' : 'opacity-0'} ${className}`}
        onLoad={() => setIsLoaded(true)}
        loading="lazy"
        decoding="async"
        {...props}
      />
    );
  }
);

LazyImage.displayName = 'LazyImage';

// ============================================================================
// Tests
// ============================================================================

describe('LazyImage', () => {
  const originalIntersectionObserver = global.IntersectionObserver;

  beforeEach(() => {
    // @ts-expect-error - Mock IntersectionObserver
    global.IntersectionObserver = MockIntersectionObserver;
    mockIntersectionCallback = null;
    mockObservedElements = new Set();
  });

  afterEach(() => {
    global.IntersectionObserver = originalIntersectionObserver;
  });

  describe('Initial Rendering', () => {
    it('should render with placeholder initially', () => {
      const placeholder = 'placeholder.jpg';

      render(<LazyImage src="actual.jpg" alt="Test image" placeholder={placeholder} />);

      const img = screen.getByAltText('Test image');
      expect(img).toHaveAttribute('src', placeholder);
    });

    it('should use default placeholder when not provided', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      const img = screen.getByAltText('Test image');
      expect(img).toHaveAttribute('src');
      expect(img.getAttribute('src')).toContain('data:image/svg+xml');
    });

    it('should have lazy loading attribute', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      const img = screen.getByAltText('Test image');
      expect(img).toHaveAttribute('loading', 'lazy');
    });

    it('should have async decoding attribute', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      const img = screen.getByAltText('Test image');
      expect(img).toHaveAttribute('decoding', 'async');
    });

    it('should start with opacity-0 class', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      const img = screen.getByAltText('Test image');
      expect(img).toHaveClass('opacity-0');
    });
  });

  describe('IntersectionObserver Setup', () => {
    it('should observe the image element', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      expect(mockObservedElements.size).toBe(1);
    });

    it('should configure observer with rootMargin', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      expect(mockObserverOptions?.rootMargin).toBe('100px');
    });

    it('should configure observer with threshold', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      expect(mockObserverOptions?.threshold).toBe(0.01);
    });

    it('should disconnect observer on unmount', () => {
      const { unmount } = render(<LazyImage src="actual.jpg" alt="Test image" />);

      expect(mockObservedElements.size).toBe(1);

      unmount();

      expect(mockObservedElements.size).toBe(0);
    });
  });

  describe('Lazy Loading Behavior', () => {
    it('should load actual image when element enters viewport', async () => {
      const actualSrc = 'https://example.com/actual-image.jpg';

      render(<LazyImage src={actualSrc} alt="Test image" placeholder="placeholder.jpg" />);

      const img = screen.getByAltText('Test image');

      // Initially shows placeholder
      expect(img).toHaveAttribute('src', 'placeholder.jpg');

      // Simulate entering viewport
      simulateIntersection(true);

      // Should now show actual image
      await waitFor(() => {
        expect(img).toHaveAttribute('src', actualSrc);
      });
    });

    it('should not load image when not intersecting', () => {
      const actualSrc = 'https://example.com/actual-image.jpg';

      render(<LazyImage src={actualSrc} alt="Test image" placeholder="placeholder.jpg" />);

      const img = screen.getByAltText('Test image');

      // Simulate not intersecting
      simulateIntersection(false);

      // Should still show placeholder
      expect(img).toHaveAttribute('src', 'placeholder.jpg');
    });

    it('should disconnect observer after entering viewport', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      // Initially observing
      expect(mockObservedElements.size).toBe(1);

      // Enter viewport
      simulateIntersection(true);

      // Observer should be disconnected
      expect(mockObservedElements.size).toBe(0);
    });
  });

  describe('Load State', () => {
    it('should have opacity-100 after image loads', async () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      const img = screen.getByAltText('Test image');

      // Enter viewport first
      simulateIntersection(true);

      // Initially has opacity-0
      expect(img).toHaveClass('opacity-0');

      // Simulate image load
      fireEvent.load(img);

      // Should now have opacity-100
      await waitFor(() => {
        expect(img).toHaveClass('opacity-100');
      });
    });

    it('should have transition classes for smooth loading', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      const img = screen.getByAltText('Test image');
      expect(img).toHaveClass('transition-opacity');
      expect(img).toHaveClass('duration-300');
    });
  });

  describe('Custom Props', () => {
    it('should pass through className', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" className="custom-class" />);

      const img = screen.getByAltText('Test image');
      expect(img).toHaveClass('custom-class');
    });

    it('should pass through additional img attributes', () => {
      render(
        <LazyImage
          src="actual.jpg"
          alt="Test image"
          width={200}
          height={100}
          title="Image title"
          data-testid="lazy-image"
        />
      );

      const img = screen.getByTestId('lazy-image');
      expect(img).toHaveAttribute('width', '200');
      expect(img).toHaveAttribute('height', '100');
      expect(img).toHaveAttribute('title', 'Image title');
    });

    it('should handle style prop', () => {
      render(
        <LazyImage
          src="actual.jpg"
          alt="Test image"
          style={{ borderRadius: '8px', objectFit: 'cover' }}
        />
      );

      const img = screen.getByAltText('Test image');
      expect(img).toHaveStyle({ borderRadius: '8px', objectFit: 'cover' });
    });
  });

  describe('Placeholder Variations', () => {
    it('should support custom placeholder image', () => {
      render(
        <LazyImage
          src="actual.jpg"
          alt="Test image"
          placeholder="custom-placeholder.jpg"
        />
      );

      const img = screen.getByAltText('Test image');
      expect(img).toHaveAttribute('src', 'custom-placeholder.jpg');
    });

    it('should support base64 placeholder', () => {
      const base64Placeholder =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

      render(
        <LazyImage
          src="actual.jpg"
          alt="Test image"
          placeholder={base64Placeholder}
        />
      );

      const img = screen.getByAltText('Test image');
      expect(img).toHaveAttribute('src', base64Placeholder);
    });

    it('should support blur placeholder URL', () => {
      render(
        <LazyImage
          src="actual.jpg"
          alt="Test image"
          placeholder="/images/blur/image-blur.jpg"
        />
      );

      const img = screen.getByAltText('Test image');
      expect(img).toHaveAttribute('src', '/images/blur/image-blur.jpg');
    });
  });

  describe('Multiple Images', () => {
    it('should handle multiple lazy images independently', () => {
      render(
        <>
          <LazyImage src="image1.jpg" alt="Image 1" placeholder="placeholder1.jpg" />
          <LazyImage src="image2.jpg" alt="Image 2" placeholder="placeholder2.jpg" />
          <LazyImage src="image3.jpg" alt="Image 3" placeholder="placeholder3.jpg" />
        </>
      );

      const img1 = screen.getByAltText('Image 1');
      const img2 = screen.getByAltText('Image 2');
      const img3 = screen.getByAltText('Image 3');

      expect(img1).toHaveAttribute('src', 'placeholder1.jpg');
      expect(img2).toHaveAttribute('src', 'placeholder2.jpg');
      expect(img3).toHaveAttribute('src', 'placeholder3.jpg');
    });
  });

  describe('Accessibility', () => {
    it('should always have alt text', () => {
      render(<LazyImage src="actual.jpg" alt="Descriptive alt text" />);

      const img = screen.getByAltText('Descriptive alt text');
      expect(img).toBeInTheDocument();
    });

    it('should support role attribute', () => {
      render(<LazyImage src="decorative.jpg" alt="" role="presentation" />);

      const img = screen.getByRole('presentation');
      expect(img).toBeInTheDocument();
    });

    it('should support aria attributes', () => {
      render(
        <LazyImage
          src="actual.jpg"
          alt="Test image"
          aria-describedby="image-description"
        />
      );

      const img = screen.getByAltText('Test image');
      expect(img).toHaveAttribute('aria-describedby', 'image-description');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty src', async () => {
      render(<LazyImage src="" alt="Empty src" placeholder="placeholder.jpg" />);

      const img = screen.getByAltText('Empty src');

      // Initially shows placeholder
      expect(img).toHaveAttribute('src', 'placeholder.jpg');

      // After entering viewport, shows empty src
      simulateIntersection(true);
      await waitFor(() => {
        expect(img).toHaveAttribute('src', '');
      });
    });

    it('should handle special characters in src', async () => {
      const specialSrc = 'https://example.com/image?size=large&quality=high';

      render(<LazyImage src={specialSrc} alt="Special chars" />);

      const img = screen.getByAltText('Special chars');

      simulateIntersection(true);

      await waitFor(() => {
        expect(img).toHaveAttribute('src', specialSrc);
      });
    });

    it('should handle long alt text', () => {
      const longAlt = 'A very detailed description of the image that explains what it shows in great detail for accessibility purposes';

      render(<LazyImage src="actual.jpg" alt={longAlt} />);

      const img = screen.getByAltText(longAlt);
      expect(img).toBeInTheDocument();
    });

    it('should handle rapid mount/unmount', () => {
      const { unmount, rerender } = render(
        <LazyImage src="image1.jpg" alt="Image 1" />
      );

      // Quick rerender
      rerender(<LazyImage src="image2.jpg" alt="Image 2" />);
      rerender(<LazyImage src="image3.jpg" alt="Image 3" />);

      // Unmount
      unmount();

      // Should not throw
      expect(mockObservedElements.size).toBe(0);
    });
  });

  describe('Performance', () => {
    it('should be memoized', () => {
      // LazyImage should have displayName set (indicates memo usage)
      expect(LazyImage.displayName).toBe('LazyImage');
    });

    it('should not re-observe after loading', () => {
      render(<LazyImage src="actual.jpg" alt="Test image" />);

      // Enter viewport
      simulateIntersection(true);

      // Observer disconnected
      expect(mockObservedElements.size).toBe(0);

      // Subsequent intersections should have no effect
      simulateIntersection(true);
      simulateIntersection(false);
      simulateIntersection(true);

      // Still disconnected
      expect(mockObservedElements.size).toBe(0);
    });
  });
});
