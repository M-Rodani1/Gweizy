/**
 * Hook to get current wallet address
 */
import { useState, useEffect } from 'react';
import { getCurrentAccount, onAccountsChanged } from '../utils/wallet';

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
      setAddress(accounts.length > 0 ? accounts[0] : null);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  return address;
}

