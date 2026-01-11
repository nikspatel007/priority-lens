import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { useAuthContext } from '@/context/AuthContext';
import { useGoogle } from '@/context/GoogleContext';
import { SignInScreen } from '@/screens/SignInScreen';
import { LandingScreen } from '@/screens/LandingScreen';
import { SyncProgressScreen } from '@/screens/SyncProgressScreen';
import { ConversationScreen } from '@/screens/ConversationScreen';
import { SettingsScreen } from '@/screens/SettingsScreen';
import { colors, typography, spacing } from '@/theme';

export type RootStackParamList = {
  SignIn: undefined;
  Landing: undefined;
  SyncProgress: undefined;
  Conversation: undefined;
  Settings: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

/**
 * RootNavigator handles the app's authentication flow
 *
 * Flow:
 * 1. Not signed in → SignInScreen
 * 2. Signed in, Google not connected → LandingScreen
 * 3. Google connected, sync in progress → SyncProgressScreen
 * 4. Google connected, sync complete → ConversationScreen
 *
 * Settings is accessible from ConversationScreen
 */
export function RootNavigator(): React.JSX.Element {
  const { isLoading: authLoading, isSignedIn } = useAuthContext();
  const { isLoading: googleLoading, isConnected: googleConnected } = useGoogle();
  const [syncComplete, setSyncComplete] = useState(false);

  /* istanbul ignore next */
  const handleSyncComplete = useCallback(() => {
    setSyncComplete(true);
  }, []);

  /* istanbul ignore next */
  const handleSyncError = useCallback((error: string) => {
    console.error('Sync error:', error);
    // Could show an alert or stay on sync screen for retry
  }, []);

  // Show loading while checking auth state
  if (authLoading) {
    return (
      <View style={styles.loadingContainer} testID="loading-container">
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
                onSkip={handleSyncComplete}
              />
            )}
          </Stack.Screen>
        ) : (
          // Sync complete - show main app screens
          <>
            <Stack.Screen name="Conversation">
              {({ navigation }) => (
                <ConversationScreen
                  onSettingsPress={() => navigation.navigate('Settings')}
                />
              )}
            </Stack.Screen>
            <Stack.Screen name="Settings">
              {({ navigation }) => (
                <SettingsScreen onBack={() => navigation.goBack()} />
              )}
            </Stack.Screen>
          </>
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
    backgroundColor: colors.backgrounds.primary,
  },
  loadingText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.lg,
    color: colors.text.secondary,
  },
});
