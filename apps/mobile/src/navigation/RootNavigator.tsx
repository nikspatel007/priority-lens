import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useAuthContext } from '@/context/AuthContext';
import { useGoogle } from '@/context/GoogleContext';
import { SignInScreen } from '@/screens/SignInScreen';
import { LandingScreen } from '@/screens/LandingScreen';
import { SyncProgressScreen } from '@/screens/SyncProgressScreen';
import { colors, typography, spacing } from '@/theme';

export type RootStackParamList = {
  SignIn: undefined;
  Landing: undefined;
  SyncProgress: undefined;
  Main: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

/**
 * Placeholder main screen - will be replaced with actual conversation screen
 */
function MainScreen(): React.JSX.Element {
  return (
    <View style={styles.mainContainer}>
      <Text style={styles.mainTitle}>Priority Lens</Text>
      <Text style={styles.mainSubtitle}>You're all set!</Text>
      <Text style={styles.mainDescription}>
        Your emails are synced and ready.{'\n'}
        Voice AI assistant coming soon.
      </Text>
    </View>
  );
}

/**
 * RootNavigator handles the app's authentication flow
 *
 * Flow:
 * 1. Not signed in → SignInScreen
 * 2. Signed in, Google not connected → LandingScreen
 * 3. Google connected, sync in progress → SyncProgressScreen
 * 4. Google connected, sync complete → MainScreen
 */
export function RootNavigator(): React.JSX.Element {
  const { isLoading: authLoading, isSignedIn } = useAuthContext();
  const { isLoading: googleLoading, isConnected: googleConnected } = useGoogle();
  const [syncComplete, setSyncComplete] = useState(false);
  const [syncStarted, setSyncStarted] = useState(false);

  const handleSyncComplete = useCallback(() => {
    setSyncComplete(true);
  }, []);

  const handleSyncError = useCallback((error: string) => {
    console.error('Sync error:', error);
    // Could show an alert or stay on sync screen for retry
  }, []);

  // Show loading while checking auth state
  if (authLoading) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.loadingText}>Loading...</Text>
      </View>
    );
  }

  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {!isSignedIn ? (
          // Not signed in - show sign in screen
          <Stack.Screen name="SignIn" component={SignInScreen} />
        ) : !googleConnected && !googleLoading ? (
          // Signed in but Google not connected
          <Stack.Screen name="Landing" component={LandingScreen} />
        ) : googleConnected && !syncComplete ? (
          // Google connected, show sync progress
          <Stack.Screen name="SyncProgress">
            {() => (
              <SyncProgressScreen
                onComplete={handleSyncComplete}
                onError={handleSyncError}
              />
            )}
          </Stack.Screen>
        ) : (
          // Sync complete - show main app
          <Stack.Screen name="Main" component={MainScreen} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.primary,
  },
  loadingText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.lg,
    color: colors.text.secondary,
  },
  mainContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.primary,
    padding: spacing[4],
  },
  mainTitle: {
    fontFamily: typography.fontFamily.serif,
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing[1],
  },
  mainSubtitle: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xl,
    color: colors.primary[500],
    marginBottom: spacing[4],
  },
  mainDescription: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.secondary,
    textAlign: 'center',
    lineHeight: typography.fontSize.base * typography.lineHeight.relaxed,
  },
});
