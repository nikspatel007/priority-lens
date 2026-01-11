import React from 'react';
import { Text } from 'react-native';
import { render, waitFor, act } from '@testing-library/react-native';
import {
  AuthProvider,
  useAuthContext,
} from '../AuthContext';
import type { AuthContextValue } from '../AuthContext';

const mockSetApiAuthTokenGetter = jest.fn();

jest.mock('@/services/api', () => ({
  setAuthTokenGetter: (...args: unknown[]) => mockSetApiAuthTokenGetter(...args),
}));

// Mock Clerk hooks
const mockSignOut = jest.fn();
const mockGetToken = jest.fn();
let mockUseAuth = jest.fn();
let mockUseUser = jest.fn();

jest.mock('@clerk/clerk-expo', () => ({
  useAuth: () => mockUseAuth(),
  useUser: () => mockUseUser(),
  ClerkProvider: ({ children }: { children: React.ReactNode }) => children,
}));

// Test component that uses the context
function TestConsumer({ onValue }: { onValue: (value: AuthContextValue) => void }): React.JSX.Element {
  const auth = useAuthContext();
  onValue(auth);
  return (
    <Text testID="test-output">
      {JSON.stringify({
        isLoading: auth.isLoading,
        isSignedIn: auth.isSignedIn,
        userId: auth.user?.id,
      })}
    </Text>
  );
}

// Helper to safely get captured value
function getCapturedValue(capturedValue: AuthContextValue | null): AuthContextValue {
  if (!capturedValue) {
    throw new Error('capturedValue is null');
  }
  return capturedValue;
}

describe('AuthContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSignOut.mockResolvedValue(undefined);
    mockGetToken.mockResolvedValue('test-token');

    // Reset to default loading state
    mockUseAuth = jest.fn().mockReturnValue({
      isLoaded: false,
      isSignedIn: false,
      signOut: mockSignOut,
      getToken: mockGetToken,
    });

    mockUseUser = jest.fn().mockReturnValue({
      user: null,
    });
  });

  describe('AuthProvider', () => {
    it('provides initial loading state', () => {
      mockUseAuth.mockReturnValue({
        isLoaded: false,
        isSignedIn: false,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      expect(value.isLoading).toBe(true);
      expect(value.isSignedIn).toBe(false);
      expect(value.user).toBeNull();
    });

    it('provides signed out state when not authenticated', () => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: false,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      expect(value.isLoading).toBe(false);
      expect(value.isSignedIn).toBe(false);
      expect(value.user).toBeNull();
    });

    it('defaults isSignedIn to false when undefined', () => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: undefined,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      expect(value.isSignedIn).toBe(false);
    });

    it('provides user when signed in', () => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: true,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      mockUseUser.mockReturnValue({
        user: {
          id: 'user-123',
          primaryEmailAddress: { emailAddress: 'test@example.com' },
          firstName: 'John',
          lastName: 'Doe',
          fullName: 'John Doe',
          imageUrl: 'https://example.com/avatar.jpg',
        },
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      expect(value.isLoading).toBe(false);
      expect(value.isSignedIn).toBe(true);
      expect(value.user).toEqual({
        id: 'user-123',
        email: 'test@example.com',
        firstName: 'John',
        lastName: 'Doe',
        fullName: 'John Doe',
        imageUrl: 'https://example.com/avatar.jpg',
      });
    });

    it('handles user with missing optional fields', () => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: true,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      mockUseUser.mockReturnValue({
        user: {
          id: 'user-456',
          primaryEmailAddress: null,
          firstName: null,
          lastName: null,
          fullName: null,
          imageUrl: null,
        },
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      expect(value.user).toEqual({
        id: 'user-456',
        email: '',
        firstName: undefined,
        lastName: undefined,
        fullName: undefined,
        imageUrl: undefined,
      });
    });

    it('returns null user when clerkUser is null', () => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: true,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      mockUseUser.mockReturnValue({
        user: null,
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      expect(value.user).toBeNull();
    });
  });

  describe('signOut', () => {
    it('calls Clerk signOut', async () => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: true,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      await act(async () => {
        await value.signOut();
      });

      expect(mockSignOut).toHaveBeenCalled();
    });
  });

  describe('getToken', () => {
    it('returns JWT token from Clerk', async () => {
      mockGetToken.mockResolvedValue('jwt-token-123');

      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: true,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      let token: string | null = null;
      await act(async () => {
        token = await value.getToken();
      });

      expect(token).toBe('jwt-token-123');
      expect(mockGetToken).toHaveBeenCalled();
    });

    it('returns null when no token available', async () => {
      mockGetToken.mockResolvedValue(null);

      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: false,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      let capturedValue: AuthContextValue | null = null;
      render(
        <AuthProvider>
          <TestConsumer onValue={(v) => { capturedValue = v; }} />
        </AuthProvider>
      );

      const value = getCapturedValue(capturedValue);
      let token: string | null = 'initial';
      await act(async () => {
        token = await value.getToken();
      });

      expect(token).toBeNull();
    });
  });

  describe('token getter', () => {
    it('sets auth token getter when loaded', async () => {
      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: true,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      render(
        <AuthProvider>
          <TestConsumer onValue={() => {}} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(mockSetApiAuthTokenGetter).toHaveBeenCalled();
      });
    });

    it('token getter returns token', async () => {
      mockGetToken.mockResolvedValue('bearer-token');

      mockUseAuth.mockReturnValue({
        isLoaded: true,
        isSignedIn: true,
        signOut: mockSignOut,
        getToken: mockGetToken,
      });

      render(
        <AuthProvider>
          <TestConsumer onValue={() => {}} />
        </AuthProvider>
      );

      await waitFor(async () => {
        expect(mockSetApiAuthTokenGetter).toHaveBeenCalled();
        const getter = mockSetApiAuthTokenGetter.mock.calls.at(-1)?.[0] as
          | (() => Promise<string | null>)
          | null
          | undefined;
        expect(typeof getter).toBe('function');
        const token = await getter?.();
        expect(token).toBe('bearer-token');
      });
    });
  });

  describe('useAuthContext', () => {
    it('throws when used outside provider', () => {
      // Suppress console.error for this test
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        render(<TestConsumer onValue={() => {}} />);
      }).toThrow('useAuthContext must be used within an AuthProvider');

      consoleSpy.mockRestore();
    });
  });
});
