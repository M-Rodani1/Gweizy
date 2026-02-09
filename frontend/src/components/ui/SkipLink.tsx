/**
 * Skip to content link for keyboard accessibility.
 *
 * Allows keyboard users to skip past navigation and jump directly
 * to the main content. Hidden visually but accessible via keyboard.
 *
 * @module components/ui/SkipLink
 */

import React from 'react';

export interface SkipLinkProps {
  /** ID of the main content element to skip to */
  targetId?: string;
  /** Link text (default: "Skip to main content") */
  children?: React.ReactNode;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Skip link component for keyboard navigation.
 *
 * @example
 * ```tsx
 * // In your App or Layout component:
 * <SkipLink targetId="main-content" />
 * <header>...</header>
 * <nav>...</nav>
 * <main id="main-content">...</main>
 * ```
 */
export const SkipLink: React.FC<SkipLinkProps> = ({
  targetId = 'main-content',
  children = 'Skip to main content',
  className = '',
}) => {
  const handleClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();
    const target = document.getElementById(targetId);
    if (target) {
      // Set tabindex to make it focusable if not already
      if (!target.hasAttribute('tabindex')) {
        target.setAttribute('tabindex', '-1');
      }
      target.focus();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <a
      href={`#${targetId}`}
      onClick={handleClick}
      className={`
        skip-link
        fixed top-0 left-0 z-[9999]
        px-4 py-2
        bg-cyan-600 text-white font-semibold
        transform -translate-y-full
        focus:translate-y-0
        transition-transform duration-200
        outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2
        ${className}
      `.trim()}
    >
      {children}
    </a>
  );
};

/**
 * Hook to manage skip link target.
 * Call this in your main content component to ensure it's focusable.
 *
 * @example
 * ```tsx
 * function MainContent() {
 *   const mainRef = useSkipLinkTarget();
 *   return <main ref={mainRef} id="main-content">...</main>;
 * }
 * ```
 */
export function useSkipLinkTarget() {
  const ref = React.useRef<HTMLElement>(null);

  React.useEffect(() => {
    const element = ref.current;
    if (element && !element.hasAttribute('tabindex')) {
      element.setAttribute('tabindex', '-1');
    }
  }, []);

  return ref;
}

export default SkipLink;
