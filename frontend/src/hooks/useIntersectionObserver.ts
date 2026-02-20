/**
 * Intersection Observer hook for viewport visibility tracking.
 *
 * Useful for:
 * - Lazy loading content
 * - Analytics (tracking when elements come into view)
 * - Infinite scroll
 * - Animation triggers
 *
 * @module hooks/useIntersectionObserver
 */

import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Options for the intersection observer.
 */
export interface UseIntersectionObserverOptions {
  /** Element that is used as the viewport */
  root?: Element | null;
  /** Margin around the root */
  rootMargin?: string;
  /** Percentage of element visible to trigger (0-1) */
  threshold?: number | number[];
  /** Only trigger once when element enters viewport */
  triggerOnce?: boolean;
  /** Callback when element enters viewport */
  onEnter?: (entry: IntersectionObserverEntry) => void;
  /** Callback when element leaves viewport */
  onLeave?: (entry: IntersectionObserverEntry) => void;
  /** Whether to observe (default: true) */
  enabled?: boolean;
}

/**
 * Return type for useIntersectionObserver hook.
 */
export interface UseIntersectionObserverReturn<T extends Element> {
  /** Ref to attach to the observed element */
  ref: React.RefObject<T | null>;
  /** Whether the element is currently in viewport */
  isIntersecting: boolean;
  /** The intersection entry object */
  entry: IntersectionObserverEntry | null;
  /** Manually unobserve the element */
  unobserve: () => void;
}

/**
 * Hook for observing element visibility in the viewport.
 *
 * @param options - Observer configuration
 * @returns Object with ref, visibility state, and controls
 *
 * @example
 * ```tsx
 * function LazyImage({ src, alt }) {
 *   const { ref, isIntersecting } = useIntersectionObserver({
 *     triggerOnce: true,
 *     rootMargin: '200px',
 *   });
 *
 *   return (
 *     <div ref={ref}>
 *       {isIntersecting ? (
 *         <img src={src} alt={alt} />
 *       ) : (
 *         <div className="placeholder" />
 *       )}
 *     </div>
 *   );
 * }
 * ```
 *
 * @example
 * ```tsx
 * function AnalyticsSection({ sectionId }) {
 *   const { ref } = useIntersectionObserver({
 *     threshold: 0.5,
 *     onEnter: () => analytics.track('section_viewed', { sectionId }),
 *   });
 *
 *   return <section ref={ref}>...</section>;
 * }
 * ```
 */
export function useIntersectionObserver<T extends Element = HTMLDivElement>(
  options: UseIntersectionObserverOptions = {}
): UseIntersectionObserverReturn<T> {
  const {
    root = null,
    rootMargin = '0px',
    threshold = 0,
    triggerOnce = false,
    onEnter,
    onLeave,
    enabled = true,
  } = options;

  const ref = useRef<T>(null);
  const [entry, setEntry] = useState<IntersectionObserverEntry | null>(null);
  const [isIntersecting, setIsIntersecting] = useState(false);
  const hasTriggered = useRef(false);
  const observerRef = useRef<IntersectionObserver | null>(null);

  const unobserve = useCallback(() => {
    if (observerRef.current && ref.current) {
      observerRef.current.unobserve(ref.current);
    }
  }, []);

  useEffect(() => {
    if (!enabled || !ref.current) {
      return;
    }

    // Check if already triggered in triggerOnce mode
    if (triggerOnce && hasTriggered.current) {
      return;
    }

    // Create observer
    const observerCallback: IntersectionObserverCallback = (entries) => {
      const [observerEntry] = entries;
      setEntry(observerEntry);
      setIsIntersecting(observerEntry.isIntersecting);

      if (observerEntry.isIntersecting) {
        onEnter?.(observerEntry);

        if (triggerOnce) {
          hasTriggered.current = true;
          unobserve();
        }
      } else {
        onLeave?.(observerEntry);
      }
    };

    observerRef.current = new IntersectionObserver(observerCallback, {
      root,
      rootMargin,
      threshold,
    });

    observerRef.current.observe(ref.current);

    return () => {
      observerRef.current?.disconnect();
    };
  }, [enabled, root, rootMargin, threshold, triggerOnce, onEnter, onLeave, unobserve]);

  return {
    ref,
    isIntersecting,
    entry,
    unobserve,
  };
}

/**
 * Hook for tracking when multiple elements enter the viewport.
 * Useful for tracking impressions across a list of items.
 *
 * @param options - Observer configuration
 * @returns Object with register function and visibility map
 *
 * @example
 * ```tsx
 * function ProductList({ products }) {
 *   const { register, visibleIds } = useMultipleIntersectionObserver({
 *     onEnter: (id) => analytics.track('product_viewed', { productId: id }),
 *   });
 *
 *   return (
 *     <div>
 *       {products.map(product => (
 *         <div key={product.id} ref={register(product.id)}>
 *           {product.name}
 *         </div>
 *       ))}
 *     </div>
 *   );
 * }
 * ```
 */
export function useMultipleIntersectionObserver<T = string>(
  options: Omit<UseIntersectionObserverOptions, 'onEnter' | 'onLeave'> & {
    onEnter?: (id: T) => void;
    onLeave?: (id: T) => void;
  } = {}
) {
  const { root = null, rootMargin = '0px', threshold = 0, triggerOnce = false, onEnter, onLeave, enabled = true } =
    options;

  const [visibleIds, setVisibleIds] = useState<Set<T>>(new Set());
  const elementsRef = useRef<Map<T, Element>>(new Map());
  const triggeredRef = useRef<Set<T>>(new Set());
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Create/update observer
  useEffect(() => {
    if (!enabled) return;

    const observerCallback: IntersectionObserverCallback = (entries) => {
      entries.forEach((entry) => {
        // Find the ID for this element
        let entryId: T | undefined;
        elementsRef.current.forEach((element, id) => {
          if (element === entry.target) {
            entryId = id;
          }
        });

        if (entryId === undefined) return;

        if (entry.isIntersecting) {
          if (triggerOnce && triggeredRef.current.has(entryId)) {
            return;
          }

          setVisibleIds((prev) => new Set(prev).add(entryId as T));
          onEnter?.(entryId);

          if (triggerOnce) {
            triggeredRef.current.add(entryId);
          }
        } else {
          setVisibleIds((prev) => {
            const next = new Set(prev);
            next.delete(entryId as T);
            return next;
          });
          onLeave?.(entryId);
        }
      });
    };

    observerRef.current = new IntersectionObserver(observerCallback, {
      root,
      rootMargin,
      threshold,
    });

    // Observe all registered elements
    elementsRef.current.forEach((element) => {
      observerRef.current?.observe(element);
    });

    return () => {
      observerRef.current?.disconnect();
    };
  }, [enabled, root, rootMargin, threshold, triggerOnce, onEnter, onLeave]);

  // Register function that returns a ref callback
  const register = useCallback(
    (id: T) => (element: Element | null) => {
      if (element) {
        elementsRef.current.set(id, element);
        observerRef.current?.observe(element);
      } else {
        const existingElement = elementsRef.current.get(id);
        if (existingElement) {
          observerRef.current?.unobserve(existingElement);
          elementsRef.current.delete(id);
        }
      }
    },
    []
  );

  const isVisible = useCallback((id: T) => visibleIds.has(id), [visibleIds]);

  return {
    register,
    visibleIds,
    isVisible,
  };
}

/**
 * Simple hook to check if an element is in viewport.
 *
 * @param options - Observer options
 * @returns Tuple of [ref, isInViewport]
 *
 * @example
 * ```tsx
 * function Component() {
 *   const [ref, isInViewport] = useInViewport();
 *
 *   return (
 *     <div ref={ref}>
 *       {isInViewport ? 'Visible!' : 'Not visible'}
 *     </div>
 *   );
 * }
 * ```
 */
export function useInViewport<T extends Element = HTMLDivElement>(
  options: Omit<UseIntersectionObserverOptions, 'onEnter' | 'onLeave'> = {}
): [React.RefObject<T | null>, boolean] {
  const { ref, isIntersecting } = useIntersectionObserver<T>(options);
  return [ref, isIntersecting];
}

export default useIntersectionObserver;
