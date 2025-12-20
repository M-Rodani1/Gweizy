/**
 * Progressive loading utility
 * Loads critical data first, then secondary data
 */

/**
 * Priority levels for data loading
 */
export enum LoadPriority {
  CRITICAL = 1,
  HIGH = 2,
  MEDIUM = 3,
  LOW = 4
}

interface LoadTask {
  id: string;
  priority: LoadPriority;
  loadFn: () => Promise<any>;
  onComplete?: (data: any) => void;
  onError?: (error: Error) => void;
}

class ProgressiveLoader {
  private tasks: LoadTask[] = [];
  private loading = false;

  /**
   * Add a task to the loading queue
   */
  addTask(task: LoadTask): void {
    this.tasks.push(task);
    this.tasks.sort((a, b) => a.priority - b.priority);
  }

  /**
   * Start loading tasks in priority order
   */
  async load(): Promise<void> {
    if (this.loading) {
      return;
    }

    this.loading = true;

    try {
      // Load tasks in priority order
      for (const task of this.tasks) {
        try {
          const data = await task.loadFn();
          task.onComplete?.(data);
        } catch (error) {
          const err = error instanceof Error ? error : new Error(String(error));
          task.onError?.(err);
        }
      }
    } finally {
      this.loading = false;
      this.tasks = [];
    }
  }

  /**
   * Clear all tasks
   */
  clear(): void {
    this.tasks = [];
  }
}

export const progressiveLoader = new ProgressiveLoader();
