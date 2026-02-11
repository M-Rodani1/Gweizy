export type Token<T> = string;

type Provider<T> = () => T;

export class DIContainer {
  private providers = new Map<Token<any>, Provider<any>>();
  private singletons = new Map<Token<any>, any>();

  register<T>(token: Token<T>, provider: Provider<T>, options: { singleton?: boolean } = {}) {
    this.providers.set(token, () => {
      if (!options.singleton) return provider();
      if (!this.singletons.has(token)) {
        this.singletons.set(token, provider());
      }
      return this.singletons.get(token);
    });
  }

  resolve<T>(token: Token<T>): T {
    const provider = this.providers.get(token);
    if (!provider) {
      throw new Error(`No provider registered for token: ${token}`);
    }
    return provider();
  }

  clear() {
    this.providers.clear();
    this.singletons.clear();
  }
}

export const diContainer = new DIContainer();
