import React from 'react';
import { render, act, waitFor } from '@testing-library/react-native';
import { View, Text } from 'react-native';
import {
  GoogleProvider,
  useGoogle,
  getGoogleUser,
  type GoogleContextValue,
} from '../GoogleContext';
import * as googleAuth from '@/services/googleAuth';
import * as api from '@/services/api';

// Mock googleAuth service
jest.mock('@/services/googleAuth', () => ({
  configureGoogleSignIn: jest.fn(),
  signInWithGoogle: jest.fn(),
  silentSignIn: jest.fn(),
  signOutGoogle: jest.fn(),
  revokeGoogleAccess: jest.fn(),
  getGoogleTokens: jest.fn(),
  getCurrentGoogleUser: jest.fn(),
}));

// Mock API service
jest.mock('@/services/api', () => ({
  getGmailConnectionStatus: jest.fn(),
}));

const mockGoogleAuth = googleAuth as jest.Mocked<typeof googleAuth>;
const mockApi = api as jest.Mocked<typeof api>;

const mockUser = {
  id: 'google-user-123',
  name: 'John Doe',
  email: 'john@gmail.com',
  photo: 'https://example.com/photo.jpg',
  familyName: 'Doe',
  givenName: 'John',
  idToken: 'mock-id-token',
  serverAuthCode: 'mock-server-auth-code',
};

// Test consumer component
function TestConsumer({
  onContext,
}: {
  onContext?: (ctx: GoogleContextValue) => void;
}): React.JSX.Element {
  const context = useGoogle();
  React.useEffect(() => {
    onContext?.(context);
  }, [context, onContext]);
  return (
    <View testID="consumer">
      <Text testID="isLoading">{String(context.isLoading)}</Text>
      <Text testID="isConnected">{String(context.isConnected)}</Text>
      <Text testID="error">{context.error || 'null'}</Text>
    </View>
  );
}

describe('GoogleContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default to no previous session
    mockGoogleAuth.silentSignIn.mockResolvedValue({
      success: false,
      error: 'cancelled',
      message: 'No previous session found',
    });
    // Default to backend not connected
    mockApi.getGmailConnectionStatus.mockResolvedValue({
      is_connected: false,
    });
  });

  describe('GoogleProvider', () => {
    it('configures GoogleSignIn on mount', async () => {
      render(
        <GoogleProvider webClientId="web-client-id">
          <View />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(mockGoogleAuth.configureGoogleSignIn).toHaveBeenCalledWith(
          'web-client-id',
          undefined
        );
      });
    });

    it('configures with iosClientId when provided', async () => {
      render(
        <GoogleProvider webClientId="web-client-id" iosClientId="ios-client-id">
          <View />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(mockGoogleAuth.configureGoogleSignIn).toHaveBeenCalledWith(
          'web-client-id',
          'ios-client-id'
        );
      });
    });

    it('attempts silent sign-in after configuration', async () => {
      render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(mockGoogleAuth.silentSignIn).toHaveBeenCalled();
      });
    });

    it('sets isConnected and user on successful silent sign-in when backend is connected', async () => {
      mockGoogleAuth.silentSignIn.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });
      // Backend confirms connection
      mockApi.getGmailConnectionStatus.mockResolvedValue({
        is_connected: true,
      });

      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isConnected').props.children).toBe('true');
        expect(getByTestId('isLoading').props.children).toBe('false');
      });
    });

    it('sets isConnected to false when silent sign-in succeeds but backend is not connected', async () => {
      mockGoogleAuth.silentSignIn.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });
      // Backend says not connected (needs fresh sign-in)
      mockApi.getGmailConnectionStatus.mockResolvedValue({
        is_connected: false,
      });

      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isConnected').props.children).toBe('false');
        expect(getByTestId('isLoading').props.children).toBe('false');
      });
    });

    it('sets isConnected to false when backend check fails', async () => {
      mockGoogleAuth.silentSignIn.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });
      // Backend check fails (network error)
      mockApi.getGmailConnectionStatus.mockRejectedValue(new Error('Network error'));

      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isConnected').props.children).toBe('false');
        expect(getByTestId('isLoading').props.children).toBe('false');
      });
    });

    it('does not call onConnectionComplete on silent sign-in (only on connect)', async () => {
      const onConnectionComplete = jest.fn().mockResolvedValue(undefined);
      mockGoogleAuth.silentSignIn.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });
      mockApi.getGmailConnectionStatus.mockResolvedValue({
        is_connected: true,
      });

      render(
        <GoogleProvider
          webClientId="web-client-id"
          onConnectionComplete={onConnectionComplete}
        >
          <TestConsumer />
        </GoogleProvider>
      );

      // Wait for silent sign-in to complete
      await waitFor(() => {
        expect(mockGoogleAuth.silentSignIn).toHaveBeenCalled();
      });

      // Give time for callback to be called (if it would be)
      await act(async () => {
        await new Promise((r) => setTimeout(r, 100));
      });

      // onConnectionComplete is NOT called during silent sign-in
      expect(onConnectionComplete).not.toHaveBeenCalled();
    });

    it('sets isConnected to false on failed silent sign-in', async () => {
      mockGoogleAuth.silentSignIn.mockResolvedValue({
        success: false,
        error: 'cancelled',
        message: 'No previous session found',
      });

      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isConnected').props.children).toBe('false');
        expect(getByTestId('isLoading').props.children).toBe('false');
      });
    });
  });

  describe('connect', () => {
    it('calls signInWithGoogle and updates state on success', async () => {
      mockGoogleAuth.signInWithGoogle.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });

      let contextValue: GoogleContextValue | undefined;
      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer
            onContext={(ctx) => {
              contextValue = ctx;
            }}
          />
        </GoogleProvider>
      );

      // Wait for initial render
      await waitFor(() => {
        expect(getByTestId('isLoading').props.children).toBe('false');
      });

      // Call connect
      await act(async () => {
        await contextValue?.connect();
      });

      await waitFor(() => {
        expect(mockGoogleAuth.signInWithGoogle).toHaveBeenCalled();
        expect(getByTestId('isConnected').props.children).toBe('true');
      });
    });

    it('calls onConnectionComplete on successful sign-in with server auth code', async () => {
      const onConnectionComplete = jest.fn().mockResolvedValue(undefined);
      mockGoogleAuth.signInWithGoogle.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });

      let contextValue: GoogleContextValue | undefined;
      const { getByTestId } = render(
        <GoogleProvider
          webClientId="web-client-id"
          onConnectionComplete={onConnectionComplete}
        >
          <TestConsumer
            onContext={(ctx) => {
              contextValue = ctx;
            }}
          />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isLoading').props.children).toBe('false');
      });

      await act(async () => {
        await contextValue?.connect();
      });

      await waitFor(() => {
        expect(onConnectionComplete).toHaveBeenCalledWith('mock-server-auth-code');
      });
    });

    it('sets error on failed sign-in', async () => {
      mockGoogleAuth.signInWithGoogle.mockResolvedValue({
        success: false,
        error: 'unknown',
        message: 'Something went wrong',
      });

      let contextValue: GoogleContextValue | undefined;
      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer
            onContext={(ctx) => {
              contextValue = ctx;
            }}
          />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isLoading').props.children).toBe('false');
      });

      await act(async () => {
        await contextValue?.connect();
      });

      await waitFor(() => {
        expect(getByTestId('error').props.children).toBe('Something went wrong');
        expect(getByTestId('isConnected').props.children).toBe('false');
      });
    });

    it('does not change connected state on cancelled sign-in', async () => {
      // First, simulate a successful silent sign-in so we're connected
      mockGoogleAuth.silentSignIn.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });
      // Backend confirms connection
      mockApi.getGmailConnectionStatus.mockResolvedValue({
        is_connected: true,
      });

      let contextValue: GoogleContextValue | undefined;
      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer
            onContext={(ctx) => {
              contextValue = ctx;
            }}
          />
        </GoogleProvider>
      );

      // Wait for initial connected state
      await waitFor(() => {
        expect(getByTestId('isConnected').props.children).toBe('true');
      });

      // Now simulate a cancelled sign-in
      mockGoogleAuth.signInWithGoogle.mockResolvedValue({
        success: false,
        error: 'cancelled',
        message: 'User cancelled sign-in',
      });

      await act(async () => {
        await contextValue?.connect();
      });

      // Should still be connected
      await waitFor(() => {
        expect(getByTestId('isConnected').props.children).toBe('true');
      });
    });

    it('does not call onConnectionComplete when serverAuthCode is null on connect', async () => {
      const onConnectionComplete = jest.fn().mockResolvedValue(undefined);
      mockGoogleAuth.signInWithGoogle.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: null,
      });

      let contextValue: GoogleContextValue | undefined;
      const { getByTestId } = render(
        <GoogleProvider
          webClientId="web-client-id"
          onConnectionComplete={onConnectionComplete}
        >
          <TestConsumer
            onContext={(ctx) => {
              contextValue = ctx;
            }}
          />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isLoading').props.children).toBe('false');
      });

      await act(async () => {
        await contextValue?.connect();
      });

      expect(onConnectionComplete).not.toHaveBeenCalled();
    });
  });

  describe('disconnect', () => {
    it('calls revokeGoogleAccess and clears state', async () => {
      // Start connected
      mockGoogleAuth.silentSignIn.mockResolvedValue({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });
      // Backend confirms connection
      mockApi.getGmailConnectionStatus.mockResolvedValue({
        is_connected: true,
      });
      mockGoogleAuth.revokeGoogleAccess.mockResolvedValue(undefined);

      let contextValue: GoogleContextValue | undefined;
      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer
            onContext={(ctx) => {
              contextValue = ctx;
            }}
          />
        </GoogleProvider>
      );

      // Wait for connected state
      await waitFor(() => {
        expect(getByTestId('isConnected').props.children).toBe('true');
      });

      // Call disconnect
      await act(async () => {
        await contextValue?.disconnect();
      });

      await waitFor(() => {
        expect(mockGoogleAuth.revokeGoogleAccess).toHaveBeenCalled();
        expect(getByTestId('isConnected').props.children).toBe('false');
        expect(getByTestId('isLoading').props.children).toBe('false');
      });
    });
  });

  describe('refreshToken', () => {
    it('returns access token when available', async () => {
      mockGoogleAuth.getGoogleTokens.mockResolvedValue({
        accessToken: 'access-token-123',
        idToken: 'id-token-123',
      });

      let contextValue: GoogleContextValue | undefined;
      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer
            onContext={(ctx) => {
              contextValue = ctx;
            }}
          />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isLoading').props.children).toBe('false');
      });

      let token: string | null = null;
      await act(async () => {
        token = await contextValue?.refreshToken() ?? null;
      });

      expect(token).toBe('access-token-123');
    });

    it('returns null when tokens not available', async () => {
      mockGoogleAuth.getGoogleTokens.mockResolvedValue(null);

      let contextValue: GoogleContextValue | undefined;
      const { getByTestId } = render(
        <GoogleProvider webClientId="web-client-id">
          <TestConsumer
            onContext={(ctx) => {
              contextValue = ctx;
            }}
          />
        </GoogleProvider>
      );

      await waitFor(() => {
        expect(getByTestId('isLoading').props.children).toBe('false');
      });

      let token: string | null = 'initial';
      await act(async () => {
        token = await contextValue?.refreshToken() ?? null;
      });

      expect(token).toBeNull();
    });
  });

  describe('useGoogle', () => {
    it('throws error when used outside GoogleProvider', () => {
      // Suppress console.error for this test
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        render(<TestConsumer />);
      }).toThrow('useGoogle must be used within a GoogleProvider');

      consoleSpy.mockRestore();
    });
  });

  describe('getGoogleUser', () => {
    it('returns current user from google auth', () => {
      mockGoogleAuth.getCurrentGoogleUser.mockReturnValue(mockUser);

      const user = getGoogleUser();

      expect(user).toEqual(mockUser);
    });

    it('returns null when no user', () => {
      mockGoogleAuth.getCurrentGoogleUser.mockReturnValue(null);

      const user = getGoogleUser();

      expect(user).toBeNull();
    });
  });
});
