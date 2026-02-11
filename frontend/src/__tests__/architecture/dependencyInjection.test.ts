import { describe, it, expect } from 'vitest';
import { DIContainer } from '../../utils/diContainer';

describe('dependency injection setup', () => {
  it('resolves registered providers', () => {
    const container = new DIContainer();
    container.register('answer', () => 42);

    expect(container.resolve<number>('answer')).toBe(42);
  });

  it('supports singleton providers', () => {
    const container = new DIContainer();
    let counter = 0;
    container.register('singleton', () => ({ value: counter += 1 }), { singleton: true });

    const first = container.resolve<{ value: number }>('singleton');
    const second = container.resolve<{ value: number }>('singleton');

    expect(first.value).toBe(1);
    expect(second.value).toBe(1);
  });
});
