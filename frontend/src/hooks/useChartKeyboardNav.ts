/**
 * useChartKeyboardNav - Hook for accessible keyboard navigation in charts.
 *
 * Provides keyboard navigation for data points in charts, supporting:
 * - Arrow keys (Left/Right) for navigation between points
 * - Home/End for first/last point
 * - Screen reader announcements for current point
 *
 * @module hooks/useChartKeyboardNav
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';

/**
 * Data point with label and value for keyboard navigation.
 */
export interface ChartDataPoint {
  /** Display label for the point (e.g., timestamp, category) */
  label: string;
  /** Formatted value to announce (e.g., "45.2 gwei") */
  value: string;
  /** Raw numeric value for calculations */
  rawValue?: number;
}

/**
 * Options for chart keyboard navigation.
 */
export interface UseChartKeyboardNavOptions {
  /** Data points to navigate through */
  dataPoints: ChartDataPoint[];
  /** Callback when focused point changes */
  onPointChange?: (index: number, point: ChartDataPoint) => void;
  /** Chart label for announcements */
  chartLabel?: string;
  /** Enable wrap-around navigation */
  wrap?: boolean;
}

/**
 * Return type for useChartKeyboardNav hook.
 */
export interface UseChartKeyboardNavReturn {
  /** Currently focused data point index (-1 if none) */
  focusedIndex: number;
  /** Set focused index manually */
  setFocusedIndex: (index: number) => void;
  /** Currently focused data point */
  focusedPoint: ChartDataPoint | null;
  /** Ref to attach to the chart container */
  containerRef: React.RefObject<HTMLDivElement | null>;
  /** Key handler for the container */
  handleKeyDown: (event: React.KeyboardEvent) => void;
  /** Focus handler to track when chart is active */
  handleFocus: () => void;
  /** Blur handler to reset focus state */
  handleBlur: (event: React.FocusEvent) => void;
  /** Whether the chart container is focused */
  isChartFocused: boolean;
  /** Announcement text for screen readers */
  announcement: string;
  /** Props to spread on the chart container */
  containerProps: {
    ref: React.RefObject<HTMLDivElement | null>;
    tabIndex: number;
    role: string;
    'aria-label': string;
    'aria-activedescendant': string | undefined;
    onKeyDown: (event: React.KeyboardEvent) => void;
    onFocus: () => void;
    onBlur: (event: React.FocusEvent) => void;
  };
  /** Get props for individual data point elements */
  getPointProps: (index: number) => {
    id: string;
    role: string;
    'aria-selected': boolean;
    'aria-label': string;
    tabIndex: number;
  };
}

/**
 * Hook for accessible keyboard navigation in charts.
 *
 * @example
 * ```tsx
 * const dataPoints = data.map(d => ({
 *   label: d.timestamp,
 *   value: `${d.value.toFixed(2)} gwei`,
 *   rawValue: d.value
 * }));
 *
 * const {
 *   containerProps,
 *   getPointProps,
 *   focusedIndex,
 *   announcement
 * } = useChartKeyboardNav({
 *   dataPoints,
 *   chartLabel: 'Gas price chart'
 * });
 *
 * return (
 *   <div {...containerProps}>
 *     <LiveRegion>{announcement}</LiveRegion>
 *     <svg>
 *       {data.map((point, i) => (
 *         <circle
 *           key={i}
 *           {...getPointProps(i)}
 *           cx={...}
 *           cy={...}
 *           className={focusedIndex === i ? 'focused' : ''}
 *         />
 *       ))}
 *     </svg>
 *   </div>
 * );
 * ```
 */
export function useChartKeyboardNav({
  dataPoints,
  onPointChange,
  chartLabel = 'Chart',
  wrap = true,
}: UseChartKeyboardNavOptions): UseChartKeyboardNavReturn {
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const [isChartFocused, setIsChartFocused] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const chartId = useRef(`chart-${Math.random().toString(36).slice(2, 9)}`);

  const pointCount = dataPoints.length;

  // Get current focused point
  const focusedPoint = useMemo(() => {
    if (focusedIndex >= 0 && focusedIndex < pointCount) {
      return dataPoints[focusedIndex];
    }
    return null;
  }, [focusedIndex, dataPoints, pointCount]);

  // Announce point change
  const announcePoint = useCallback((index: number) => {
    if (index >= 0 && index < pointCount) {
      const point = dataPoints[index];
      const position = `${index + 1} of ${pointCount}`;
      setAnnouncement(`${point.label}: ${point.value}. Point ${position}.`);
    }
  }, [dataPoints, pointCount]);

  // Move to next/previous point
  const moveFocus = useCallback((direction: 'next' | 'prev' | 'first' | 'last') => {
    if (pointCount === 0) return;

    let newIndex: number;

    switch (direction) {
      case 'next':
        if (focusedIndex === -1) {
          newIndex = 0;
        } else if (focusedIndex >= pointCount - 1) {
          newIndex = wrap ? 0 : pointCount - 1;
        } else {
          newIndex = focusedIndex + 1;
        }
        break;
      case 'prev':
        if (focusedIndex === -1) {
          newIndex = pointCount - 1;
        } else if (focusedIndex <= 0) {
          newIndex = wrap ? pointCount - 1 : 0;
        } else {
          newIndex = focusedIndex - 1;
        }
        break;
      case 'first':
        newIndex = 0;
        break;
      case 'last':
        newIndex = pointCount - 1;
        break;
      default:
        return;
    }

    setFocusedIndex(newIndex);
    announcePoint(newIndex);
    onPointChange?.(newIndex, dataPoints[newIndex]);
  }, [focusedIndex, pointCount, wrap, announcePoint, onPointChange, dataPoints]);

  // Key handler
  const handleKeyDown = useCallback((event: React.KeyboardEvent) => {
    if (pointCount === 0) return;

    switch (event.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        event.preventDefault();
        moveFocus('next');
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        event.preventDefault();
        moveFocus('prev');
        break;
      case 'Home':
        event.preventDefault();
        moveFocus('first');
        break;
      case 'End':
        event.preventDefault();
        moveFocus('last');
        break;
      case 'Escape':
        event.preventDefault();
        setFocusedIndex(-1);
        setAnnouncement('');
        break;
    }
  }, [pointCount, moveFocus]);

  // Focus handler
  const handleFocus = useCallback(() => {
    setIsChartFocused(true);
    if (focusedIndex === -1 && pointCount > 0) {
      // Auto-focus first point when entering chart
      setFocusedIndex(0);
      announcePoint(0);
    }
  }, [focusedIndex, pointCount, announcePoint]);

  // Blur handler
  const handleBlur = useCallback((event: React.FocusEvent) => {
    // Only blur if focus left the container entirely
    if (!containerRef.current?.contains(event.relatedTarget as Node)) {
      setIsChartFocused(false);
      // Keep focusedIndex so user can return to same point
    }
  }, []);

  // Reset focus when data changes significantly
  useEffect(() => {
    if (focusedIndex >= pointCount) {
      setFocusedIndex(pointCount > 0 ? pointCount - 1 : -1);
    }
  }, [pointCount, focusedIndex]);

  // Container props
  const containerProps = useMemo(() => ({
    ref: containerRef,
    tabIndex: 0,
    role: 'application',
    'aria-label': `${chartLabel}. Use arrow keys to navigate ${pointCount} data points.`,
    'aria-activedescendant': focusedIndex >= 0 ? `${chartId.current}-point-${focusedIndex}` : undefined,
    onKeyDown: handleKeyDown,
    onFocus: handleFocus,
    onBlur: handleBlur,
  }), [chartLabel, pointCount, focusedIndex, handleKeyDown, handleFocus, handleBlur]);

  // Get props for individual data points
  const getPointProps = useCallback((index: number) => {
    const point = dataPoints[index];
    const isSelected = focusedIndex === index;

    return {
      id: `${chartId.current}-point-${index}`,
      role: 'option',
      'aria-selected': isSelected,
      'aria-label': point ? `${point.label}: ${point.value}` : '',
      tabIndex: -1, // Points are not directly focusable
    };
  }, [dataPoints, focusedIndex]);

  return {
    focusedIndex,
    setFocusedIndex,
    focusedPoint,
    containerRef,
    handleKeyDown,
    handleFocus,
    handleBlur,
    isChartFocused,
    announcement,
    containerProps,
    getPointProps,
  };
}

export default useChartKeyboardNav;
