import { describe, it, expect } from 'vitest';
import { createStateMachine } from '../../utils/stateMachine';

describe('state machine patterns', () => {
  it('transitions between states', () => {
    const machine = createStateMachine('idle', {
      idle: { start: 'loading' },
      loading: { succeed: 'success', fail: 'error' },
      error: { retry: 'loading' }
    });

    expect(machine.getState()).toBe('idle');
    expect(machine.can('start')).toBe(true);
    machine.transition('start');
    expect(machine.getState()).toBe('loading');
    machine.transition('succeed');
    expect(machine.getState()).toBe('success');
  });

  it('throws on invalid transitions', () => {
    const machine = createStateMachine('idle', { idle: { start: 'loading' } });
    expect(() => machine.transition('fail')).toThrow('Invalid transition');
  });
});
