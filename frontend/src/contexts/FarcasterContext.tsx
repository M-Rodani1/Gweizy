import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import sdk from '@farcaster/miniapp-sdk';

interface FarcasterContextType {
  isSDKLoaded: boolean;
  context: any | null;
  user: any | null;
  error: string | null;
  openUrl: (url: string) => void;
  signMessage: (message: string) => Promise<string | null>;
}

const FarcasterContext = createContext<FarcasterContextType | null>(null);

export const useFarcaster = () => {
  const context = useContext(FarcasterContext);
  if (!context) {
    // Return default values if not in Farcaster context
    return {
      isSDKLoaded: false,
      context: null,
      user: null,
      error: null,
      openUrl: (url: string) => window.open(url, '_blank'),
      signMessage: async () => null
    };
  }
  return context;
};

interface FarcasterProviderProps {
  children: ReactNode;
}

export const FarcasterProvider: React.FC<FarcasterProviderProps> = ({ children }) => {
  const [isSDKLoaded, setIsSDKLoaded] = useState(false);
  const [context, setContext] = useState<any | null>(null);
  const [user, setUser] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initFarcaster = async () => {
      try {
        // Check if we're in a Farcaster miniapp environment
        const isInFarcaster = window.parent !== window;

        if (!isInFarcaster) {
          console.log('Not running in Farcaster miniapp - SDK features disabled');
          setIsSDKLoaded(false);
          return;
        }

        // Initialize the SDK
        await sdk.actions.ready();

        // Get context
        const farcasterContext = await sdk.context;
        setContext(farcasterContext);

        // Get user info if available
        if (farcasterContext?.user) {
          setUser(farcasterContext.user);
          console.log('Farcaster user:', farcasterContext.user);
        }

        setIsSDKLoaded(true);
        console.log('Farcaster SDK loaded successfully');

      } catch (err) {
        console.error('Failed to initialize Farcaster SDK:', err);
        setError(err instanceof Error ? err.message : 'Failed to initialize Farcaster SDK');
        setIsSDKLoaded(false);
      }
    };

    initFarcaster();
  }, []);

  const openUrl = (url: string) => {
    if (isSDKLoaded && sdk.actions.openUrl) {
      sdk.actions.openUrl(url);
    } else {
      window.open(url, '_blank');
    }
  };

  const signMessage = async (message: string): Promise<string | null> => {
    if (!isSDKLoaded || !sdk.actions.signMessage) {
      console.warn('Farcaster SDK not available for signing');
      return null;
    }

    try {
      const signature = await sdk.actions.signMessage({ message });
      return signature;
    } catch (err) {
      console.error('Failed to sign message:', err);
      return null;
    }
  };

  const value: FarcasterContextType = {
    isSDKLoaded,
    context,
    user,
    error,
    openUrl,
    signMessage
  };

  return (
    <FarcasterContext.Provider value={value}>
      {children}
    </FarcasterContext.Provider>
  );
};
