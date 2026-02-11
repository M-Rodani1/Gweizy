export interface OfflineAction<TPayload = unknown> {
  type: string;
  payload: TPayload;
}

const STORAGE_KEY = 'gweizy_offline_queue';

const readQueue = (): OfflineAction[] => {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as OfflineAction[];
  } catch {
    return [];
  }
};

const writeQueue = (queue: OfflineAction[]) => {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(queue));
  } catch {
    // ignore storage errors
  }
};

export const enqueueOfflineAction = (action: OfflineAction) => {
  const queue = readQueue();
  queue.push(action);
  writeQueue(queue);
};

export const flushOfflineQueue = async (handler: (action: OfflineAction) => Promise<void>) => {
  const queue = readQueue();
  for (const action of queue) {
    await handler(action);
  }
  writeQueue([]);
};

export const getOfflineQueue = (): OfflineAction[] => readQueue();
