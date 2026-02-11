import { describe, it, expect } from 'vitest';
import { CommandBus } from '../../utils/commandBus';

describe('command pattern for actions', () => {
  it('registers and executes commands', async () => {
    const bus = new CommandBus();
    bus.register({
      type: 'increment',
      execute: (value: number) => value + 1
    });

    const result = await bus.execute<number, number>('increment', 1);
    expect(result).toBe(2);
  });

  it('throws when command is missing', async () => {
    const bus = new CommandBus();
    await expect(bus.execute('missing', {})).rejects.toThrow('Command not registered');
  });
});
