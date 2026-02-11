export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3
};

let currentLevel: LogLevel = (import.meta.env.DEV ? 'debug' : 'info');

export const setLogLevel = (level: LogLevel) => {
  currentLevel = level;
};

export const getLogLevel = (): LogLevel => currentLevel;

const shouldLog = (level: LogLevel) => LEVELS[level] >= LEVELS[currentLevel];

const formatMessage = (message: string, context?: Record<string, unknown>) => {
  if (!context) return message;
  return `${message} ${JSON.stringify(context)}`;
};

export const logDebug = (message: string, context?: Record<string, unknown>) => {
  if (shouldLog('debug')) console.debug(formatMessage(message, context));
};

export const logInfo = (message: string, context?: Record<string, unknown>) => {
  if (shouldLog('info')) console.info(formatMessage(message, context));
};

export const logWarn = (message: string, context?: Record<string, unknown>) => {
  if (shouldLog('warn')) console.warn(formatMessage(message, context));
};

export const logError = (message: string, context?: Record<string, unknown>) => {
  if (shouldLog('error')) console.error(formatMessage(message, context));
};

export const logEvent = (
  level: LogLevel,
  message: string,
  context?: Record<string, unknown>
) => {
  switch (level) {
    case 'debug':
      logDebug(message, context);
      break;
    case 'info':
      logInfo(message, context);
      break;
    case 'warn':
      logWarn(message, context);
      break;
    case 'error':
      logError(message, context);
      break;
  }
};
