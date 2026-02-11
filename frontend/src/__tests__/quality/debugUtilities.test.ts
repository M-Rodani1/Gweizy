import { describe, it, expect, vi, beforeEach } from 'vitest';
import { enableDebug, disableDebug, isDebugEnabled, debugLog } from '../../utils/debug';
import * as logger from '../../utils/logger';

describe('debug utilities', () => {
  beforeEach(() => {
    disableDebug();
  });

  it('toggles debug mode', () => {
    expect(isDebugEnabled()).toBe(false);
    enableDebug();
    expect(isDebugEnabled()).toBe(true);
    disableDebug();
    expect(isDebugEnabled()).toBe(false);
  });

  it('logs only when debug is enabled', () => {
    const spy = vi.spyOn(logger, 'logDebug').mockImplementation(() => {});

    debugLog('skip');
    expect(spy).not.toHaveBeenCalled();

    enableDebug();
    debugLog('run', { source: 'test' });
    expect(spy).toHaveBeenCalledWith('run', { source: 'test' });
  });
});
