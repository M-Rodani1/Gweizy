import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import WalletConnect from '../../components/WalletConnect';

const walletMocks = vi.hoisted(() => ({
  isWalletInstalled: vi.fn(),
  connectWallet: vi.fn(),
  getCurrentAccount: vi.fn(),
  getCurrentChainId: vi.fn(),
  formatAddress: vi.fn((address: string) => address),
  copyToClipboard: vi.fn(),
  getBaseScanAddressUrl: vi.fn((address: string) => `https://example.com/${address}`),
  onAccountsChanged: vi.fn(() => () => {}),
  onChainChanged: vi.fn(() => () => {}),
}));

vi.mock('../../utils/wallet', () => walletMocks);

describe('touch target sizing', () => {
  beforeEach(() => {
    walletMocks.isWalletInstalled.mockReset();
    walletMocks.connectWallet.mockReset();
    walletMocks.getCurrentAccount.mockReset();
    walletMocks.getCurrentChainId.mockReset();
    walletMocks.formatAddress.mockReset();
    walletMocks.copyToClipboard.mockReset();
    walletMocks.getBaseScanAddressUrl.mockReset();
    walletMocks.onAccountsChanged.mockReset();
    walletMocks.onChainChanged.mockReset();

    walletMocks.formatAddress.mockImplementation((address: string) => address);
    walletMocks.onAccountsChanged.mockReturnValue(() => {});
    walletMocks.onChainChanged.mockReturnValue(() => {});
  });

  it('ensures primary wallet actions meet minimum touch target size', async () => {
    walletMocks.isWalletInstalled.mockReturnValue(true);
    walletMocks.getCurrentAccount.mockResolvedValue(null);
    walletMocks.getCurrentChainId.mockResolvedValue(8453);

    render(<WalletConnect />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /connect wallet/i })).toBeInTheDocument();
    });

    const button = screen.getByRole('button', { name: /connect wallet/i });
    expect(button.className).toContain('min-h-[44px]');
  });

  it('ensures install wallet link meets minimum touch target size', () => {
    walletMocks.isWalletInstalled.mockReturnValue(false);

    render(<WalletConnect />);

    const link = screen.getByRole('link', { name: /install wallet/i });
    expect(link.className).toContain('min-h-[44px]');
  });
});
