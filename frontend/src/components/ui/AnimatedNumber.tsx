import React, { useEffect, useRef, useState } from 'react';

interface AnimatedNumberProps {
  value: number;
  /** Number of decimal places to show */
  decimals?: number;
  /** Duration of animation in ms */
  duration?: number;
  /** Prefix to show before number (e.g., "$") */
  prefix?: string;
  /** Suffix to show after number (e.g., " gwei") */
  suffix?: string;
  /** Additional CSS classes */
  className?: string;
  /** Whether to flash on change */
  flash?: boolean;
  /** Format as compact (e.g., 1.2K, 3.4M) */
  compact?: boolean;
}

/**
 * Animated number counter with smooth transitions
 * Automatically animates between value changes
 */
const AnimatedNumber: React.FC<AnimatedNumberProps> = ({
  value,
  decimals = 2,
  duration = 500,
  prefix = '',
  suffix = '',
  className = '',
  flash = true,
  compact = false,
}) => {
  const [displayValue, setDisplayValue] = useState(value);
  const [isAnimating, setIsAnimating] = useState(false);
  const previousValue = useRef(value);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    // Skip animation on initial render or if value hasn't changed
    if (previousValue.current === value) {
      return;
    }

    const startValue = previousValue.current;
    const endValue = value;
    const startTime = performance.now();

    setIsAnimating(true);

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Easing function (ease-out cubic)
      const easeOut = 1 - Math.pow(1 - progress, 3);

      const currentValue = startValue + (endValue - startValue) * easeOut;
      setDisplayValue(currentValue);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        setDisplayValue(endValue);
        setIsAnimating(false);
        previousValue.current = endValue;
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [value, duration]);

  // Format number
  const formatNumber = (num: number): string => {
    if (compact) {
      if (num >= 1_000_000) {
        return (num / 1_000_000).toFixed(1) + 'M';
      }
      if (num >= 1_000) {
        return (num / 1_000).toFixed(1) + 'K';
      }
    }
    return num.toFixed(decimals);
  };

  const formattedValue = formatNumber(displayValue);

  return (
    <span
      className={`
        inline-block tabular-nums
        ${flash && isAnimating ? 'number-flash' : ''}
        ${className}
      `}
    >
      {prefix}{formattedValue}{suffix}
    </span>
  );
};

/**
 * Hook for using animated numbers with more control
 */
export function useAnimatedNumber(
  targetValue: number,
  duration: number = 500
): number {
  const [value, setValue] = useState(targetValue);
  const previousValue = useRef(targetValue);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    if (previousValue.current === targetValue) {
      return;
    }

    const startValue = previousValue.current;
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);

      const currentValue = startValue + (targetValue - startValue) * easeOut;
      setValue(currentValue);

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        setValue(targetValue);
        previousValue.current = targetValue;
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [targetValue, duration]);

  return value;
}

export default AnimatedNumber;
