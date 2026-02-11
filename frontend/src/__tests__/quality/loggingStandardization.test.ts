import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { logInfo, logWarn, logDebug, logError, setLogLevel } from '../../utils/logger';

describe('logging standardization', () => {
  const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
  const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
  const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

  beforeEach(() => {
    infoSpy.mockClear();
    warnSpy.mockClear();
    debugSpy.mockClear();
    errorSpy.mockClear();
  });

  afterEach(() => {
    setLogLevel('debug');
  });

  it('respects log level thresholds', () => {
    setLogLevel('warn');

    logDebug('debug');
    logInfo('info');
    logWarn('warn');
    logError('error');

    expect(debugSpy).not.toHaveBeenCalled();
    expect(infoSpy).not.toHaveBeenCalled();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(errorSpy).toHaveBeenCalledTimes(1);
  });

  it('serializes context for structured logs', () => {
    setLogLevel('info');
    logInfo('event', { source: 'test', count: 2 });

    expect(infoSpy).toHaveBeenCalledWith('event {"source":"test","count":2}');
  });
});
