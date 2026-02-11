export type Variant = 'A' | 'B';

export interface Experiment {
  key: string;
  trafficSplit?: number; // percentage to variant B
}

const STORAGE_KEY = 'gweizy_experiments';

const hash = (value: string) => {
  let hashValue = 0;
  for (let i = 0; i < value.length; i += 1) {
    hashValue = (hashValue * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hashValue;
};

const readAssignments = (): Record<string, Variant> => {
  if (typeof window === 'undefined') return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as Record<string, Variant>;
  } catch {
    return {};
  }
};

const writeAssignments = (assignments: Record<string, Variant>) => {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(assignments));
  } catch {
    // ignore storage errors
  }
};

export const assignVariant = (experiment: Experiment, subjectId: string): Variant => {
  const assignments = readAssignments();
  const existing = assignments[experiment.key];
  if (existing) return existing;

  const split = experiment.trafficSplit ?? 50;
  const bucket = hash(`${experiment.key}:${subjectId}`) % 100;
  const variant: Variant = bucket < split ? 'B' : 'A';
  assignments[experiment.key] = variant;
  writeAssignments(assignments);
  return variant;
};

export const getAssignedVariant = (experimentKey: string): Variant | null => {
  const assignments = readAssignments();
  return assignments[experimentKey] ?? null;
};
