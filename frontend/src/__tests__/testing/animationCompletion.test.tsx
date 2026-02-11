import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { useAnimationComplete } from '../../hooks/useAnimationComplete';

function AnimatedBox({ onComplete }: { onComplete: () => void }) {
  const ref = useAnimationComplete<HTMLDivElement>(onComplete);
  return <div ref={ref} data-testid="box" />;
}

describe('animation/transition completion', () => {
  it('invokes callback on transition end', () => {
    const onComplete = vi.fn();
    const { getByTestId } = render(<AnimatedBox onComplete={onComplete} />);

    const box = getByTestId('box');
    box.dispatchEvent(new Event('transitionend'));

    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('invokes callback on animation end', () => {
    const onComplete = vi.fn();
    const { getByTestId } = render(<AnimatedBox onComplete={onComplete} />);

    const box = getByTestId('box');
    box.dispatchEvent(new Event('animationend'));

    expect(onComplete).toHaveBeenCalledTimes(1);
  });
});
