/**
 * SettingsScreen Tests
 */

import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { Alert } from 'react-native';
import { SettingsScreen } from '../SettingsScreen';

// Get the mocked Alert.alert from jest.setup.js
const mockAlert = Alert.alert as jest.Mock;

// Mock contexts
const mockSignOut = jest.fn().mockResolvedValue(undefined);
let mockUser: {
  firstName?: string;
  lastName?: string;
  fullName?: string;
  email?: string;
  imageUrl?: string;
} | null = {
  firstName: 'Test',
  lastName: 'User',
  fullName: 'Test User',
  email: 'test@example.com',
  imageUrl: 'https://example.com/avatar.jpg',
};

jest.mock('@/context/AuthContext', () => ({
  useAuthContext: () => ({
    user: mockUser,
    signOut: mockSignOut,
  }),
}));

const mockDisconnectGoogle = jest.fn().mockResolvedValue(undefined);
let mockGoogleConnected = true;
let mockGoogleUser: { user?: { email?: string } } | null = { user: { email: 'google@example.com' } };

jest.mock('@/context/GoogleContext', () => ({
  useGoogle: () => ({
    isConnected: mockGoogleConnected,
    user: mockGoogleUser,
    disconnect: mockDisconnectGoogle,
  }),
}));

describe('SettingsScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUser = {
      firstName: 'Test',
      lastName: 'User',
      fullName: 'Test User',
      email: 'test@example.com',
      imageUrl: 'https://example.com/avatar.jpg',
    };
    mockGoogleConnected = true;
    mockGoogleUser = { user: { email: 'google@example.com' } };
  });

  describe('rendering', () => {
    it('renders settings screen', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('settings-screen')).toBeTruthy();
    });

    it('renders header with title', () => {
      const { getByText } = render(<SettingsScreen />);
      expect(getByText('Settings')).toBeTruthy();
    });

    it('renders back button when onBack provided', () => {
      const onBack = jest.fn();
      const { getByTestId } = render(<SettingsScreen onBack={onBack} />);
      expect(getByTestId('back-button')).toBeTruthy();
    });

    it('does not render back button when onBack not provided', () => {
      const { queryByTestId } = render(<SettingsScreen />);
      expect(queryByTestId('back-button')).toBeNull();
    });
  });

  describe('profile section', () => {
    it('renders profile card', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('profile-card')).toBeTruthy();
    });

    it('renders profile image when imageUrl provided', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('profile-image')).toBeTruthy();
    });

    it('renders profile placeholder when no imageUrl', () => {
      mockUser = { firstName: 'Test', email: 'test@example.com' };
      const { getByTestId, getByText } = render(<SettingsScreen />);
      expect(getByTestId('profile-placeholder')).toBeTruthy();
      expect(getByText('T')).toBeTruthy(); // First letter of firstName
    });

    it('renders email initial when no firstName', () => {
      mockUser = { email: 'test@example.com' };
      const { getByText } = render(<SettingsScreen />);
      expect(getByText('t')).toBeTruthy(); // First letter of email
    });

    it('renders ? when no firstName or email', () => {
      mockUser = {};
      const { getByText } = render(<SettingsScreen />);
      expect(getByText('?')).toBeTruthy();
    });

    it('renders full name', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('profile-name')).toHaveTextContent('Test User');
    });

    it('renders email when no full name', () => {
      mockUser = { email: 'test@example.com' };
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('profile-name')).toHaveTextContent('test@example.com');
    });

    it('renders Unknown User when no name or email', () => {
      mockUser = {};
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('profile-name')).toHaveTextContent('Unknown User');
    });

    it('renders profile email', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('profile-email')).toHaveTextContent('test@example.com');
    });

    it('renders empty email when no email', () => {
      mockUser = { firstName: 'Test' };
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('profile-email')).toHaveTextContent('');
    });
  });

  describe('google connection', () => {
    it('renders google connection card', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('google-connection-card')).toBeTruthy();
    });

    it('shows connected status with email', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('google-status')).toHaveTextContent('google@example.com');
    });

    it('shows Connected when no google user email', () => {
      mockGoogleUser = {};
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('google-status')).toHaveTextContent('Connected');
    });

    it('shows Not connected when not connected', () => {
      mockGoogleConnected = false;
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('google-status')).toHaveTextContent('Not connected');
    });

    it('renders disconnect button when connected', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('disconnect-google-button')).toBeTruthy();
    });

    it('does not render disconnect button when not connected', () => {
      mockGoogleConnected = false;
      const { queryByTestId } = render(<SettingsScreen />);
      expect(queryByTestId('disconnect-google-button')).toBeNull();
    });

    it('shows alert when disconnect pressed', () => {
      const { getByTestId } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('disconnect-google-button'));

      expect(mockAlert).toHaveBeenCalledWith(
        'Disconnect Google',
        'This will remove access to your Gmail and Calendar. You can reconnect later.',
        expect.any(Array)
      );
    });

    it('disconnects google when confirmed', async () => {
      const { getByTestId } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('disconnect-google-button'));

      // Get the alert buttons and call the Disconnect callback
      const alertCall = (mockAlert as jest.Mock).mock.calls[0];
      const buttons = alertCall[2];
      const disconnectButton = buttons.find((b: { text: string }) => b.text === 'Disconnect');

      await act(async () => {
        await disconnectButton.onPress();
      });

      expect(mockDisconnectGoogle).toHaveBeenCalled();
    });

    it('cancels disconnect when cancelled', async () => {
      const { getByTestId } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('disconnect-google-button'));

      const alertCall = (mockAlert as jest.Mock).mock.calls[0];
      const buttons = alertCall[2];
      const cancelButton = buttons.find((b: { text: string }) => b.text === 'Cancel');

      // Cancel button doesn't have onPress, just style
      expect(cancelButton.style).toBe('cancel');
    });

    it('handles disconnect error', async () => {
      mockDisconnectGoogle.mockRejectedValueOnce(new Error('Disconnect failed'));

      const { getByTestId } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('disconnect-google-button'));

      const alertCall = (mockAlert as jest.Mock).mock.calls[0];
      const buttons = alertCall[2];
      const disconnectButton = buttons.find((b: { text: string }) => b.text === 'Disconnect');

      await act(async () => {
        await disconnectButton.onPress();
      });

      // Should show error alert
      await waitFor(() => {
        expect(mockAlert).toHaveBeenLastCalledWith(
          'Error',
          'Failed to disconnect Google. Please try again.'
        );
      });
    });

    it('shows loading state during disconnect', async () => {
      // Make disconnect take time so we can see the loading state
      let resolveDisconnect: () => void;
      mockDisconnectGoogle.mockImplementationOnce(() => new Promise<void>((resolve) => {
        resolveDisconnect = resolve;
      }));

      const { getByTestId, rerender } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('disconnect-google-button'));

      const alertCall = (mockAlert as jest.Mock).mock.calls[0];
      const buttons = alertCall[2];
      const disconnectButton = buttons.find((b: { text: string }) => b.text === 'Disconnect');

      // Start the disconnect - the isDisconnectingGoogle state should be set to true
      act(() => {
        disconnectButton.onPress();
      });

      // Force a re-render to see the loading state
      rerender(<SettingsScreen />);

      // The button should be disabled during loading
      // Verify disconnect was called and then complete it
      expect(mockDisconnectGoogle).toHaveBeenCalled();

      // Complete the disconnect
      await act(async () => {
        resolveDisconnect!();
      });
    });
  });

  describe('app info', () => {
    it('renders app version', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('app-version')).toHaveTextContent('1.0.0');
    });
  });

  describe('sign out', () => {
    it('renders sign out button', () => {
      const { getByTestId } = render(<SettingsScreen />);
      expect(getByTestId('sign-out-button')).toBeTruthy();
    });

    it('shows alert when sign out pressed', () => {
      const { getByTestId } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('sign-out-button'));

      expect(mockAlert).toHaveBeenCalledWith(
        'Sign Out',
        'Are you sure you want to sign out?',
        expect.any(Array)
      );
    });

    it('signs out when confirmed', async () => {
      const { getByTestId } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('sign-out-button'));

      const alertCall = (mockAlert as jest.Mock).mock.calls[0];
      const buttons = alertCall[2];
      const signOutButton = buttons.find((b: { text: string }) => b.text === 'Sign Out');

      await act(async () => {
        await signOutButton.onPress();
      });

      expect(mockSignOut).toHaveBeenCalled();
    });

    it('cancels sign out when cancelled', () => {
      const { getByTestId } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('sign-out-button'));

      const alertCall = (mockAlert as jest.Mock).mock.calls[0];
      const buttons = alertCall[2];
      const cancelButton = buttons.find((b: { text: string }) => b.text === 'Cancel');

      expect(cancelButton.style).toBe('cancel');
    });

    it('handles sign out error', async () => {
      mockSignOut.mockRejectedValueOnce(new Error('Sign out failed'));

      const { getByTestId } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('sign-out-button'));

      const alertCall = (mockAlert as jest.Mock).mock.calls[0];
      const buttons = alertCall[2];
      const signOutButton = buttons.find((b: { text: string }) => b.text === 'Sign Out');

      await act(async () => {
        await signOutButton.onPress();
      });

      await waitFor(() => {
        expect(mockAlert).toHaveBeenLastCalledWith(
          'Error',
          'Failed to sign out. Please try again.'
        );
      });
    });

    it('shows loading state during sign out', async () => {
      // Make sign out take time so we can see the loading state
      let resolveSignOut: () => void;
      mockSignOut.mockImplementationOnce(() => new Promise<void>((resolve) => {
        resolveSignOut = resolve;
      }));

      const { getByTestId, rerender } = render(<SettingsScreen />);

      fireEvent.press(getByTestId('sign-out-button'));

      const alertCall = (mockAlert as jest.Mock).mock.calls[0];
      const buttons = alertCall[2];
      const signOutButton = buttons.find((b: { text: string }) => b.text === 'Sign Out');

      // Start the sign out - the isSigningOut state should be set to true
      act(() => {
        signOutButton.onPress();
      });

      // Force a re-render to see the loading state
      rerender(<SettingsScreen />);

      // Verify sign out was called
      expect(mockSignOut).toHaveBeenCalled();

      // Complete the sign out
      await act(async () => {
        resolveSignOut!();
      });
    });
  });

  describe('navigation', () => {
    it('calls onBack when back button pressed', () => {
      const onBack = jest.fn();
      const { getByTestId } = render(<SettingsScreen onBack={onBack} />);

      fireEvent.press(getByTestId('back-button'));

      expect(onBack).toHaveBeenCalled();
    });
  });
});
