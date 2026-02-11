import { useEffect, useRef } from 'react';

export const useAnimationComplete = <T extends HTMLElement>(
  onComplete: () => void
) => {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const handler = () => onComplete();
    node.addEventListener('transitionend', handler);
    node.addEventListener('animationend', handler);

    return () => {
      node.removeEventListener('transitionend', handler);
      node.removeEventListener('animationend', handler);
    };
  }, [onComplete]);

  return ref;
};
