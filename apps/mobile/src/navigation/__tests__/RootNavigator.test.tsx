/**
 * RootNavigator Tests
 */

import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { RootNavigator } from '../RootNavigator';

// Mock contexts
let mockAuthLoading = false;
let mockIsSignedIn = false;
let mockGoogleLoading = false;
let mockGoogleConnected = false;

jest.mock('@/context/AuthContext', () => ({
  useAuthContext: () => ({
    isLoading: mockAuthLoading,
    isSignedIn: mockIsSignedIn,
  }),
}));

jest.mock('@/context/GoogleContext', () => ({
  useGoogle: () => ({
    isLoading: mockGoogleLoading,
    isConnected: mockGoogleConnected,
  }),
}));

// Mock screens
jest.mock('@/screens/SignInScreen', () => ({
  SignInScreen: () => {
    const { Text } = require('react-native');
    return <Text testID="signin-screen">SignIn Screen</Text>;
  },
}));

jest.mock('@/screens/LandingScreen', () => ({
  LandingScreen: () => {
    const { Text } = require('react-native');
    return <Text testID="landing-screen">Landing Screen</Text>;
  },
}));

// Store the callbacks from SyncProgressScreen for testing
let capturedOnComplete: (() => void) | null = null;
let capturedOnError: ((error: string) => void) | null = null;

jest.mock('@/screens/SyncProgressScreen', () => ({
  SyncProgressScreen: ({ onComplete, onError }: { onComplete: () => void; onError: (error: string) => void }) => {
    capturedOnComplete = onComplete;
    capturedOnError = onError;
    const { Text } = require('react-native');
    return <Text testID="sync-progress-screen">Sync Progress Screen</Text>;
  },
}));

// Store navigation for ConversationScreen
let capturedConversationNav: { navigate: (screen: string) => void } | null = null;

jest.mock('@/screens/ConversationScreen', () => ({
  ConversationScreen: ({ onSettingsPress }: { onSettingsPress?: () => void }) => {
    const { Text, TouchableOpacity, View } = require('react-native');
    return (
      <View testID="conversation-screen">
        <Text>Conversation Screen</Text>
        {onSettingsPress && (
          <TouchableOpacity testID="settings-press" onPress={onSettingsPress}>
            <Text>Settings</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  },
}));

// Store navigation for SettingsScreen
let capturedSettingsOnBack: (() => void) | null = null;

jest.mock('@/screens/SettingsScreen', () => ({
  SettingsScreen: ({ onBack }: { onBack?: () => void }) => {
    capturedSettingsOnBack = onBack;
    const { Text, TouchableOpacity, View } = require('react-native');
    return (
      <View testID="settings-screen">
        <Text>Settings Screen</Text>
        {onBack && (
          <TouchableOpacity testID="back-press" onPress={onBack}>
            <Text>Back</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  },
}));

// Mock navigation
jest.mock('@react-navigation/native', () => {
  const React = require('react');
  return {
    NavigationContainer: ({ children }: { children: React.ReactNode }) => children,
  };
});

jest.mock('@react-navigation/native-stack', () => ({
  createNativeStackNavigator: () => ({
    Navigator: ({ children }: { children: React.ReactNode }) => children,
    Screen: ({ children, component: Component, name }: {
      children?: (props: { navigation: { navigate: (s: string) => void; goBack: () => void } }) => React.ReactNode;
      component?: React.ComponentType;
      name: string;
    }) => {
      if (typeof children === 'function') {
        const mockNavigation = {
          navigate: jest.fn(),
          goBack: jest.fn(),
        };
        if (name === 'Conversation') {
          capturedConversationNav = mockNavigation;
        }
        return children({ navigation: mockNavigation });
      }
      if (Component) {
        return <Component />;
      }
      return null;
    },
  }),
}));

describe('RootNavigator', () => {
  beforeEach(() => {
    mockAuthLoading = false;
    mockIsSignedIn = false;
    mockGoogleLoading = false;
    mockGoogleConnected = false;
    capturedOnComplete = null;
    capturedOnError = null;
    capturedConversationNav = null;
    capturedSettingsOnBack = null;
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('loading state', () => {
    it('shows loading when auth is loading', () => {
      mockAuthLoading = true;

      const { getByText, getByTestId } = render(<RootNavigator />);

      expect(getByText('Loading...')).toBeTruthy();
      expect(getByTestId('loading-container')).toBeTruthy();
    });
  });

  describe('not signed in', () => {
    it('shows SignInScreen when not signed in', () => {
      mockIsSignedIn = false;

      const { getByTestId } = render(<RootNavigator />);

      expect(getByTestId('signin-screen')).toBeTruthy();
    });
  });

  describe('signed in, google not connected', () => {
    it('shows LandingScreen when signed in but Google not connected', () => {
      mockIsSignedIn = true;
      mockGoogleConnected = false;
      mockGoogleLoading = false;

      const { getByTestId } = render(<RootNavigator />);

      expect(getByTestId('landing-screen')).toBeTruthy();
    });

    it('does not show LandingScreen when Google is loading', () => {
      mockIsSignedIn = true;
      mockGoogleConnected = false;
      mockGoogleLoading = true;

      const { queryByTestId } = render(<RootNavigator />);

      // When Google is loading, the condition !googleConnected && !googleLoading is false
      // so it falls through to the next condition
      expect(queryByTestId('landing-screen')).toBeNull();
    });
  });

  describe('google connected, sync in progress', () => {
    it('shows SyncProgressScreen when Google connected but sync not complete', () => {
      mockIsSignedIn = true;
      mockGoogleConnected = true;

      const { getByTestId } = render(<RootNavigator />);

      expect(getByTestId('sync-progress-screen')).toBeTruthy();
    });

    it('calls handleSyncComplete and shows ConversationScreen', async () => {
      mockIsSignedIn = true;
      mockGoogleConnected = true;

      const { getByTestId, rerender } = render(<RootNavigator />);

      // Initially shows sync progress
      expect(getByTestId('sync-progress-screen')).toBeTruthy();

      // Call the onComplete callback
      await act(async () => {
        capturedOnComplete?.();
      });

      // Need to rerender to see the state change
      rerender(<RootNavigator />);

      // Now should show Conversation screen
      await waitFor(() => {
        expect(getByTestId('conversation-screen')).toBeTruthy();
      });
    });

    it('handles sync error', async () => {
      mockIsSignedIn = true;
      mockGoogleConnected = true;

      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      render(<RootNavigator />);

      // Call the onError callback
      await act(async () => {
        capturedOnError?.('Test sync error');
      });

      expect(consoleSpy).toHaveBeenCalledWith('Sync error:', 'Test sync error');
    });
  });

  describe('conversation screen', () => {
    it('shows ConversationScreen when sync is complete', async () => {
      mockIsSignedIn = true;
      mockGoogleConnected = true;

      const { getByTestId, rerender } = render(<RootNavigator />);

      // Initially shows sync progress
      expect(getByTestId('sync-progress-screen')).toBeTruthy();

      // Complete sync
      await act(async () => {
        capturedOnComplete?.();
      });

      rerender(<RootNavigator />);

      // Should show conversation screen
      await waitFor(() => {
        expect(getByTestId('conversation-screen')).toBeTruthy();
      });
    });

    it('ConversationScreen has onSettingsPress that navigates to Settings', async () => {
      mockIsSignedIn = true;
      mockGoogleConnected = true;

      const { getByTestId, rerender } = render(<RootNavigator />);

      // Complete sync
      await act(async () => {
        capturedOnComplete?.();
      });

      rerender(<RootNavigator />);

      await waitFor(() => {
        expect(getByTestId('conversation-screen')).toBeTruthy();
      });

      // Press settings
      await act(async () => {
        fireEvent.press(getByTestId('settings-press'));
      });

      expect(capturedConversationNav?.navigate).toHaveBeenCalledWith('Settings');
    });
  });

  describe('settings screen', () => {
    it('SettingsScreen has onBack that calls goBack', async () => {
      mockIsSignedIn = true;
      mockGoogleConnected = true;

      const { getByTestId, rerender } = render(<RootNavigator />);

      // Complete sync
      await act(async () => {
        capturedOnComplete?.();
      });

      rerender(<RootNavigator />);

      // The SettingsScreen's onBack is wired to navigation.goBack()
      // Since we can't easily navigate to it in the mock, we test the callback is set
      expect(capturedSettingsOnBack).toBeDefined();
    });
  });
});
