import React, {
  createContext,
  useContext,
  useCallback,
  useMemo,
  useState,
  useEffect,
  type ReactNode,
} from 'react';
import type { User as GoogleUser } from '@react-native-google-signin/google-signin';
import {
  configureGoogleSignIn,
  signInWithGoogle,
  silentSignIn,
  revokeGoogleAccess,
  getGoogleTokens,
  getCurrentGoogleUser,
} from '@/services/googleAuth';
import { getGmailConnectionStatus } from '@/services/api';

/**
 * Google connection state
 */
export interface GoogleConnectionState {
  /** Whether the Google SDK is loading */
  isLoading: boolean;
  /** Whether connected to Google */
  isConnected: boolean;
  /** Google user data */
  user: GoogleUser | null;
  /** Error message if connection failed */
  error: string | null;
}

/**
 * GoogleContext value interface
 */
export interface GoogleContextValue extends GoogleConnectionState {
  /** Initiate Google connection */
  connect: () => Promise<void>;
  /** Disconnect from Google */
  disconnect: () => Promise<void>;
  /** Refresh the access token */
  refreshToken: () => Promise<string | null>;
}

const GoogleContext = createContext<GoogleContextValue | null>(null);

interface GoogleProviderProps {
  children: ReactNode;
  /** Web client ID for Google OAuth */
  webClientId: string;
  /** iOS client ID for Google OAuth (optional) */
  iosClientId?: string;
  /** Callback when connection is complete (for backend sync) */
  onConnectionComplete?: (serverAuthCode: string) => Promise<void>;
}

/**
 * GoogleProvider manages Google OAuth connection state
 *
 * Features:
 * - Automatic silent sign-in on mount
 * - Connect/disconnect methods
 * - Token refresh
 * - Backend sync callback
 */
export function GoogleProvider({
  children,
  webClientId,
  iosClientId,
  onConnectionComplete,
}: GoogleProviderProps): React.JSX.Element {
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [user, setUser] = useState<GoogleUser | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConfigured, setIsConfigured] = useState(false);

  // Configure Google Sign-In on mount
  useEffect(() => {
    configureGoogleSignIn(webClientId, iosClientId);
    setIsConfigured(true);
  }, [webClientId, iosClientId]);

  // Attempt silent sign-in after configuration
  // Also verify backend has valid tokens - if not, require fresh sign-in
  useEffect(() => {
    if (!isConfigured) return;

    const attemptSilentSignIn = async () => {
      setIsLoading(true);
      setError(null);

      const result = await silentSignIn();

      if (result.success) {
        setUser(result.user);

        // Check if backend has valid tokens
        try {
          const backendStatus = await getGmailConnectionStatus();

          if (backendStatus.is_connected) {
            // Backend has tokens, we're good
            setIsConnected(true);
          } else {
            // Backend doesn't have tokens - need fresh sign-in
            // Silent sign-in doesn't give us serverAuthCode
            console.log('Backend not connected, need fresh sign-in');
            setIsConnected(false);
          }
        } catch {
          // Can't reach backend or not authenticated yet
          // Don't set connected until we verify with backend
          setIsConnected(false);
        }
      } else {
        setUser(null);
        setIsConnected(false);
      }

      setIsLoading(false);
    };

    attemptSilentSignIn();
  }, [isConfigured]);

  const connect = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    setError(null);

    const result = await signInWithGoogle();

    if (result.success) {
      setUser(result.user);
      setIsConnected(true);
      setError(null);

      // Sync with backend using server auth code
      if (onConnectionComplete && result.serverAuthCode) {
        await onConnectionComplete(result.serverAuthCode);
      }
    } else {
      setError(result.message);
      // Don't change connected state on cancel
      if (result.error !== 'cancelled') {
        setIsConnected(false);
        setUser(null);
      }
    }

    setIsLoading(false);
  }, [onConnectionComplete]);

  const disconnect = useCallback(async (): Promise<void> => {
    setIsLoading(true);

    await revokeGoogleAccess();

    setUser(null);
    setIsConnected(false);
    setError(null);
    setIsLoading(false);
  }, []);

  const refreshToken = useCallback(async (): Promise<string | null> => {
    const tokens = await getGoogleTokens();
    return tokens?.accessToken ?? null;
  }, []);

  const value: GoogleContextValue = useMemo(
    () => ({
      isLoading,
      isConnected,
      user,
      error,
      connect,
      disconnect,
      refreshToken,
    }),
    [isLoading, isConnected, user, error, connect, disconnect, refreshToken]
  );

  return (
    <GoogleContext.Provider value={value}>{children}</GoogleContext.Provider>
  );
}

/**
 * Hook to access Google context
 *
 * @throws Error if used outside GoogleProvider
 */
export function useGoogle(): GoogleContextValue {
  const context = useContext(GoogleContext);
  if (!context) {
    throw new Error('useGoogle must be used within a GoogleProvider');
  }
  return context;
}

/**
 * Get the current Google user synchronously
 *
 * Returns null if not signed in or outside provider.
 */
export function getGoogleUser(): GoogleUser | null {
  return getCurrentGoogleUser();
}
