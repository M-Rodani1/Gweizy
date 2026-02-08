/**
 * Tests for useWalletAddress hook
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { useWalletAddress } from '../../hooks/useWalletAddress';

// Track the callback for account changes
let accountChangeCallback: ((accounts: string[]) => void) | null = null;

// Mock wallet utilities
vi.mock('../../utils/wallet', () => ({
  getCurrentAccount: vi.fn(),
  onAccountsChanged: vi.fn((callback) => {
    accountChangeCallback = callback;
    return () => {
      accountChangeCallback = null;
    };
  })
}));

import { getCurrentAccount, onAccountsChanged } from '../../utils/wallet';

describe('useWalletAddress', () => {
  beforeEach(() => {
    accountChangeCallback = null;
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should return null initially before wallet check completes', () => {
    (getCurrentAccount as ReturnType<typeof vi.fn>).mockImplementation(
      () => new Promise(() => {})
    );

    const { result } = renderHook(() => useWalletAddress());

    expect(result.current).toBeNull();
  });

  it('should return connected wallet address', async () => {
    const mockAddress = '0x1234567890123456789012345678901234567890';
    (getCurrentAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockAddress);

    const { result } = renderHook(() => useWalletAddress());

    await waitFor(() => {
      expect(result.current).toBe(mockAddress);
    });
  });

  it('should return null when no wallet is connected', async () => {
    (getCurrentAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

    const { result } = renderHook(() => useWalletAddress());

    await waitFor(() => {
      expect(getCurrentAccount).toHaveBeenCalled();
    });

    expect(result.current).toBeNull();
  });

  it('should subscribe to account changes on mount', async () => {
    (getCurrentAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

    renderHook(() => useWalletAddress());

    await waitFor(() => {
      expect(onAccountsChanged).toHaveBeenCalled();
    });
  });

  it('should update address when account changes', async () => {
    const initialAddress = '0x1111111111111111111111111111111111111111';
    const newAddress = '0x2222222222222222222222222222222222222222';

    (getCurrentAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce(initialAddress);

    const { result } = renderHook(() => useWalletAddress());

    await waitFor(() => {
      expect(result.current).toBe(initialAddress);
    });

    // Simulate account change
    act(() => {
      if (accountChangeCallback) {
        accountChangeCallback([newAddress]);
      }
    });

    expect(result.current).toBe(newAddress);
  });

  it('should set null when wallet disconnects', async () => {
    const initialAddress = '0x1111111111111111111111111111111111111111';

    (getCurrentAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce(initialAddress);

    const { result } = renderHook(() => useWalletAddress());

    await waitFor(() => {
      expect(result.current).toBe(initialAddress);
    });

    // Simulate disconnect (empty accounts array)
    act(() => {
      if (accountChangeCallback) {
        accountChangeCallback([]);
      }
    });

    expect(result.current).toBeNull();
  });

  it('should unsubscribe from account changes on unmount', async () => {
    (getCurrentAccount as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

    const { unmount } = renderHook(() => useWalletAddress());

    await waitFor(() => {
      expect(onAccountsChanged).toHaveBeenCalled();
    });

    expect(accountChangeCallback).not.toBeNull();

    unmount();

    expect(accountChangeCallback).toBeNull();
  });

});

