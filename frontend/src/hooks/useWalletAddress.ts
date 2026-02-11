/**
 * Hook to get and track the current connected wallet address.
 *
 * Integrates with MetaMask/Web3 wallets to provide the current
 * account address and automatically updates when the user switches
 * accounts.
 *
 * @module hooks/useWalletAddress
 */
import { useState, useEffect } from 'react';
import { getCurrentAccount, onAccountsChanged } from '../utils/wallet';
import { isValidAddress, toChecksumAddress } from '../utils/walletAddress';

/**
 * Hook to track the currently connected wallet address.
 *
 * Automatically detects the connected MetaMask account on mount
 * and subscribes to account change events. Returns null if no
 * wallet is connected or if MetaMask is not installed.
 *
 * @returns {string|null} The connected wallet address (checksummed) or null
 *
 * @example
 * ```tsx
 * function WalletStatus() {
 *   const address = useWalletAddress();
 *
 *   if (!address) {
 *     return <button onClick={connectWallet}>Connect Wallet</button>;
 *   }
 *
 *   return (
 *     <div>
 *       Connected: {address.slice(0, 6)}...{address.slice(-4)}
 *     </div>
 *   );
 * }
 * ```
 *
 * @example
 * ```tsx
 * // Use with user-specific data fetching
 * function UserDashboard() {
 *   const address = useWalletAddress();
 *   const { data: history } = useQuery({
 *     queryKey: ['history', address],
 *     queryFn: () => fetchUserHistory(address!),
 *     enabled: !!address,
 *   });
 *
 *   return <TransactionList transactions={history} />;
 * }
 * ```
 */
export function useWalletAddress(): string | null {
  const [address, setAddress] = useState<string | null>(null);

  useEffect(() => {
    const checkConnection = async () => {
      const account = await getCurrentAccount();
      setAddress(account);
    };

    checkConnection();

    // Listen for account changes
    const unsubscribe = onAccountsChanged((accounts) => {
      if (accounts.length === 0) {
        setAddress(null);
        return;
      }

      const next = accounts[0];
      setAddress(isValidAddress(next) ? toChecksumAddress(next) : null);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  return address;
}
