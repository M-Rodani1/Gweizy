/**
 * UI Component barrel exports
 *
 * Re-exports all UI components for easier imports:
 * import { Button, Card, Badge } from '@/components/ui';
 *
 * @module components/ui
 */

// Core UI primitives
export { Button } from './Button';
export { Card } from './Card';
export { Badge } from './Badge';
export { Stat } from './Stat';
export { Chip } from './Chip';
export { SectionHeader } from './SectionHeader';
export { default as FormField } from './FormField';
export { default as Modal } from './Modal';
export { default as Combobox } from './Combobox';
export { default as CollapsibleSection } from './CollapsibleSection';
export { default as ThemeToggle } from './ThemeToggle';

// Loading states
export {
  default as Skeleton,
  SkeletonCard,
  SkeletonChart,
  SkeletonHeatmap,
  SkeletonList,
  SkeletonTable,
  SkeletonMetrics,
  SkeletonGasPrediction,
  SkeletonMultiChain,
  SkeletonGasHero,
  SkeletonForecast,
  ErrorFallback
} from './Skeleton';

// Visualization
export { default as Sparkline } from './Sparkline';
export { default as ConfidenceRing } from './ConfidenceRing';
export { default as ConfidenceBar } from './ConfidenceBar';
export { default as AnimatedNumber } from './AnimatedNumber';
export { default as NetworkPulse } from './NetworkPulse';
export { default as CorrectionBadge } from './CorrectionBadge';

// Performance
export { default as LazyImage } from './LazyImage';
export { default as VirtualizedList } from './VirtualizedList';

// Accessibility
export { SkipLink, useSkipLinkTarget } from './SkipLink';
export {
  LiveRegion,
  LiveRegionProvider,
  StatusAnnouncer,
  useAnnounceChange,
  useAnnounceError,
  useLiveRegion,
} from './LiveRegion';

// Error handling
export {
  RetryableQuery,
  RetryButton,
  MutationError,
  InlineError,
} from './RetryableQuery';

// Deprecated - use Badge instead
export { Pill } from './Pill';
