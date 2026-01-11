import { StatusBar } from 'expo-status-bar';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { ClerkProvider, ClerkLoaded } from '@clerk/clerk-expo';
import * as WebBrowser from 'expo-web-browser';
import { AuthProvider } from '@/context/AuthContext';
import { GoogleProvider } from '@/context/GoogleContext';
import { LiveKitProvider } from '@/context/LiveKitContext';
import { RootNavigator } from '@/navigation/RootNavigator';
import { tokenCache } from '@/context/TokenCache';
import { completeGoogleConnection } from '@/services/api';
import { ErrorBoundary } from '@/components/error/ErrorBoundary';
import { StyleSheet } from 'react-native';

// Required for Clerk OAuth flows (Expo WebBrowser)
WebBrowser.maybeCompleteAuthSession();

// Environment variables
const CLERK_PUBLISHABLE_KEY =
  process.env['EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY'] || '';
const GOOGLE_WEB_CLIENT_ID =
  process.env['EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID'] || '';
const GOOGLE_IOS_CLIENT_ID = process.env['EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID'];

/**
 * Handle Google connection completion
 *
 * When the user connects their Google account, send the server auth code
 * to the backend to exchange for tokens and start email sync.
 */
async function handleGoogleConnection(serverAuthCode: string): Promise<void> {
  try {
    const result = await completeGoogleConnection(serverAuthCode);
    console.log('Google connection result:', result);
    // The SyncProgressScreen will pick up from here by polling status
  } catch (error) {
    console.error('Failed to complete Google connection:', error);
    throw error;
  }
}

/**
 * Priority Lens Mobile App
 *
 * Root component that sets up:
 * - GestureHandlerRootView (required for react-native-gesture-handler)
 * - SafeAreaProvider (for safe area insets)
 * - ClerkProvider (Clerk authentication)
 * - AuthProvider (wraps Clerk hooks)
 * - GoogleProvider (Google OAuth)
 * - RootNavigator (navigation based on auth state)
 */
export default function App(): React.JSX.Element {
  // Validate required env vars
  if (!CLERK_PUBLISHABLE_KEY) {
    console.warn('Missing EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY');
  }
  if (!GOOGLE_WEB_CLIENT_ID) {
    console.warn('Missing EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID');
  }

  return (
    <GestureHandlerRootView style={styles.root}>
      <SafeAreaProvider>
        <ErrorBoundary
          onError={(error, errorInfo) => {
            console.error('App Error:', error.message);
            console.error('Component Stack:', errorInfo.componentStack);
          }}
        >
          <ClerkProvider
            publishableKey={CLERK_PUBLISHABLE_KEY}
            tokenCache={tokenCache}
          >
            <ClerkLoaded>
              <AuthProvider>
                <GoogleProvider
                  webClientId={GOOGLE_WEB_CLIENT_ID}
                  iosClientId={GOOGLE_IOS_CLIENT_ID}
                  onConnectionComplete={handleGoogleConnection}
                >
                  <LiveKitProvider>
                    <RootNavigator />
                    <StatusBar style="auto" />
                  </LiveKitProvider>
                </GoogleProvider>
              </AuthProvider>
            </ClerkLoaded>
          </ClerkProvider>
        </ErrorBoundary>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
});
