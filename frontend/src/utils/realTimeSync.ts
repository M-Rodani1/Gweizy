export type SyncCallback<T> = (payload: T) => void;

export class RealTimeSync<T> {
  private subscribers = new Set<SyncCallback<T>>();

  subscribe(callback: SyncCallback<T>) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  publish(payload: T) {
    this.subscribers.forEach((callback) => callback(payload));
  }

  clear() {
    this.subscribers.clear();
  }
}
