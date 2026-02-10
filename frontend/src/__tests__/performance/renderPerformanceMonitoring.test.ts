import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

describe('render performance monitoring', () => {
  const originalEnv = process.env.NODE_ENV;

  beforeEach(() => {
    process.env.NODE_ENV = 'development';
  });

  afterEach(() => {
    process.env.NODE_ENV = originalEnv;
  });

  it('records a metric when render exceeds threshold', async () => {
    vi.resetModules();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const perfModule = await import('../../utils/performanceOptimizations');
    perfModule.clearRenderMetrics();

    const nowSpy = vi.spyOn(globalThis.performance, 'now');
    nowSpy.mockReturnValueOnce(0).mockReturnValueOnce(25);

    const stop = perfModule.createRenderMonitor('SlowComponent', 16);
    stop();

    const metrics = perfModule.getRenderMetrics();
    expect(metrics).toHaveLength(1);
    expect(metrics[0].componentName).toBe('SlowComponent');
    expect(warnSpy).toHaveBeenCalled();

    warnSpy.mockRestore();
  });

  it('does not record metric when render is under threshold', async () => {
    vi.resetModules();
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const perfModule = await import('../../utils/performanceOptimizations');
    perfModule.clearRenderMetrics();

    const nowSpy = vi.spyOn(globalThis.performance, 'now');
    nowSpy.mockReturnValueOnce(0).mockReturnValueOnce(10);

    const stop = perfModule.createRenderMonitor('FastComponent', 16);
    stop();

    const metrics = perfModule.getRenderMetrics();
    expect(metrics).toHaveLength(0);
    expect(warnSpy).not.toHaveBeenCalled();

    warnSpy.mockRestore();
  });
});
