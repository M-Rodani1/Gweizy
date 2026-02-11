export interface MicroFrontendApp {
  name: string;
  mount: (container: HTMLElement) => void | (() => void);
}

const registry = new Map<string, MicroFrontendApp>();

export const registerMicroFrontend = (app: MicroFrontendApp) => {
  registry.set(app.name, app);
};

export const mountMicroFrontend = (name: string, container: HTMLElement) => {
  const app = registry.get(name);
  if (!app) {
    throw new Error(`Micro-frontend not registered: ${name}`);
  }
  return app.mount(container);
};

export const listMicroFrontends = () => Array.from(registry.values());
