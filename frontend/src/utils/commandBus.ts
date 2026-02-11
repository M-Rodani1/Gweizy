export interface Command<TPayload = void, TResult = void> {
  type: string;
  execute: (payload: TPayload) => TResult | Promise<TResult>;
}

export class CommandBus {
  private registry = new Map<string, Command<any, any>>();

  register(command: Command) {
    this.registry.set(command.type, command);
  }

  has(type: string) {
    return this.registry.has(type);
  }

  async execute<TPayload, TResult>(type: string, payload: TPayload): Promise<TResult> {
    const command = this.registry.get(type) as Command<TPayload, TResult> | undefined;
    if (!command) {
      throw new Error(`Command not registered: ${type}`);
    }
    return await command.execute(payload);
  }
}
