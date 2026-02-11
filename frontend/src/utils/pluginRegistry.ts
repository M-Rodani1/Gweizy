export interface Plugin {
  name: string;
  setup: () => void;
}

export class PluginRegistry {
  private plugins: Plugin[] = [];

  register(plugin: Plugin) {
    this.plugins.push(plugin);
  }

  initializeAll() {
    this.plugins.forEach((plugin) => plugin.setup());
  }

  list() {
    return [...this.plugins];
  }
}

export const pluginRegistry = new PluginRegistry();
