export interface DevToolsHandle {
  version: string;
  ping: () => string;
}

const DEVTOOLS_KEY = '__GWEIZY_DEVTOOLS__';

export const registerDevTools = (version: string): DevToolsHandle | null => {
  if (!import.meta.env.DEV || typeof window === 'undefined') {
    return null;
  }

  const handle: DevToolsHandle = {
    version,
    ping: () => 'pong'
  };

  (window as typeof window & { [DEVTOOLS_KEY]?: DevToolsHandle })[DEVTOOLS_KEY] = handle;
  return handle;
};

export const getDevTools = (): DevToolsHandle | null => {
  if (typeof window === 'undefined') return null;
  return (window as typeof window & { [DEVTOOLS_KEY]?: DevToolsHandle })[DEVTOOLS_KEY] || null;
};
