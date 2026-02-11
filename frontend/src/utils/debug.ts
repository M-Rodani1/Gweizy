import { logDebug } from './logger';

let enabled = false;

export const enableDebug = () => {
  enabled = true;
};

export const disableDebug = () => {
  enabled = false;
};

export const isDebugEnabled = () => enabled;

export const debugLog = (message: string, context?: Record<string, unknown>) => {
  if (!enabled) return;
  logDebug(message, context);
};
