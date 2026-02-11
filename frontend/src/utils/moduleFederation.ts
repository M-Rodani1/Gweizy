export interface RemoteContainer {
  init: (shareScope: unknown) => Promise<void> | void;
  get: (module: string) => Promise<() => unknown>;
}

export const loadRemoteModule = async <T>(
  scope: string,
  module: string,
  shareScope: unknown = {}
): Promise<T> => {
  if (typeof window === 'undefined') {
    throw new Error('Module federation requires a browser environment');
  }

  const container = (window as typeof window & Record<string, RemoteContainer>)[scope];
  if (!container) {
    throw new Error(`Remote container not found: ${scope}`);
  }

  await container.init(shareScope);
  const factory = await container.get(module);
  return factory() as T;
};
