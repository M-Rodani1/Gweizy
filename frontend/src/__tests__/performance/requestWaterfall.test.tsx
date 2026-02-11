import { describe, it, expect, afterEach, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import AccuracyDashboard from '../../components/AccuracyDashboard';

describe('Request waterfall elimination', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('kicks off metrics and trends fetches in parallel', async () => {
    const pending = new Promise<Response>(() => {});
    const fetchMock = vi.fn()
      .mockReturnValueOnce(pending)
      .mockReturnValueOnce(pending);

    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    await act(async () => {
      render(<AccuracyDashboard />);
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
