import React, {
  createContext,
  useContext,
  useCallback,
  useMemo,
  type ReactNode,
} from 'react';
import { useAuth, useUser } from '@clerk/clerk-expo';
import { setAuthTokenGetter as setApiAuthTokenGetter } from '@/services/api';

/**
 * User data exposed by AuthContext
 */
export interface AuthUser {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  fullName?: string;
  imageUrl?: string;
}

/**
 * AuthContext value interface
 */
export interface AuthContextValue {
  /** Whether auth state is still loading */
  isLoading: boolean;
  /** Whether user is signed in */
  isSignedIn: boolean;
  /** Current user data, null if not signed in */
  user: AuthUser | null;
  /** Sign out the current user */
  signOut: () => Promise<void>;
  /** Get JWT token for API calls */
  getToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

/**
 * AuthProvider wraps the app with authentication context
 *
 * Uses Clerk for authentication and exposes:
 * - isLoading: Whether auth state is loading
 * - isSignedIn: Whether user is authenticated
 * - user: Current user data
 * - signOut: Sign out function
 * - getToken: Get JWT token for API calls
 */
export function AuthProvider({ children }: AuthProviderProps): React.JSX.Element {
  const { isLoaded, isSignedIn, signOut: clerkSignOut, getToken: clerkGetToken } = useAuth();
  const { user: clerkUser } = useUser();

  const isLoading = !isLoaded;

  const user: AuthUser | null = useMemo(() => {
    if (!isSignedIn || !clerkUser) {
      return null;
    }

    const primaryEmail = clerkUser.primaryEmailAddress?.emailAddress;

    return {
      id: clerkUser.id,
      email: primaryEmail ?? '',
      firstName: clerkUser.firstName ?? undefined,
      lastName: clerkUser.lastName ?? undefined,
      fullName: clerkUser.fullName ?? undefined,
      imageUrl: clerkUser.imageUrl ?? undefined,
    };
  }, [isSignedIn, clerkUser]);

  const signOut = useCallback(async (): Promise<void> => {
    await clerkSignOut();
  }, [clerkSignOut]);

  const getToken = useCallback(async (): Promise<string | null> => {
    // Prefer the JWT template used by the backend (see docs/CLERK_SETUP.md).
    // Fall back to the default token if the template isn't available.
    try {
      const templated = await clerkGetToken({ template: 'priority-lens-api' });
      if (templated) return templated;
    } catch {
      // ignore
    }
    return clerkGetToken();
  }, [clerkGetToken]);

  // Set the token getter for API service when auth loads
  React.useEffect(() => {
    setApiAuthTokenGetter(isLoaded ? getToken : null);
  }, [isLoaded, getToken]);

  const value: AuthContextValue = useMemo(
    () => ({
      isLoading,
      isSignedIn: isSignedIn ?? false,
      user,
      signOut,
      getToken,
    }),
    [isLoading, isSignedIn, user, signOut, getToken]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * Hook to access auth context
 *
 * @throws Error if used outside AuthProvider
 */
export function useAuthContext(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuthContext must be used within an AuthProvider');
  }
  return context;
}
