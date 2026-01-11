import {
  GoogleSignin,
  statusCodes,
  type User as GoogleUser,
} from '@react-native-google-signin/google-signin';

/**
 * Google OAuth scopes required for Priority Lens
 */
export const GOOGLE_SCOPES = [
  'https://www.googleapis.com/auth/gmail.readonly',
  'https://www.googleapis.com/auth/calendar.readonly',
] as const;

/**
 * Result of a successful Google sign-in
 */
export interface GoogleSignInResult {
  success: true;
  user: GoogleUser;
  idToken: string | null;
  serverAuthCode: string | null;
}

/**
 * Result of a failed or cancelled Google sign-in
 */
export interface GoogleSignInError {
  success: false;
  error: 'cancelled' | 'in_progress' | 'play_services_unavailable' | 'unknown';
  message: string;
}

export type GoogleSignInResponse = GoogleSignInResult | GoogleSignInError;

/**
 * Configure Google Sign-In with the required scopes and client IDs
 *
 * Must be called once at app startup, before any sign-in attempts.
 */
export function configureGoogleSignIn(
  webClientId: string,
  iosClientId?: string
): void {
  GoogleSignin.configure({
    webClientId,
    iosClientId,
    scopes: [...GOOGLE_SCOPES],
    offlineAccess: true,
    forceCodeForRefreshToken: true,
  });
}

/**
 * Initiate Google Sign-In flow
 *
 * Opens the native Google sign-in UI and returns the result.
 */
export async function signInWithGoogle(): Promise<GoogleSignInResponse> {
  try {
    await GoogleSignin.hasPlayServices({ showPlayServicesUpdateDialog: true });
    const response = await GoogleSignin.signIn();

    // Handle the response based on the type field
    if (response.type === 'cancelled') {
      return {
        success: false,
        error: 'cancelled',
        message: 'Sign-in was cancelled by user',
      };
    }

    // Success case - extract user data
    const { data } = response;
    if (!data) {
      return {
        success: false,
        error: 'unknown',
        message: 'No user data returned from Google',
      };
    }

    return {
      success: true,
      user: data,
      idToken: data.idToken ?? null,
      serverAuthCode: data.serverAuthCode ?? null,
    };
  } catch (error) {
    return handleSignInError(error);
  }
}

/**
 * Attempt silent sign-in to restore previous session
 *
 * Returns the current user if already signed in, or attempts
 * to restore a previous session without showing UI.
 */
export async function silentSignIn(): Promise<GoogleSignInResponse> {
  try {
    const isSignedIn = await GoogleSignin.hasPreviousSignIn();
    if (!isSignedIn) {
      return {
        success: false,
        error: 'cancelled',
        message: 'No previous session found',
      };
    }

    const response = await GoogleSignin.signInSilently();

    if (response.type === 'noSavedCredentialFound') {
      return {
        success: false,
        error: 'cancelled',
        message: 'No saved credentials found',
      };
    }

    const { data } = response;
    if (!data) {
      return {
        success: false,
        error: 'unknown',
        message: 'No user data returned from silent sign-in',
      };
    }

    return {
      success: true,
      user: data,
      idToken: data.idToken ?? null,
      serverAuthCode: data.serverAuthCode ?? null,
    };
  } catch (error) {
    return handleSignInError(error);
  }
}

/**
 * Sign out from Google
 */
export async function signOutGoogle(): Promise<void> {
  try {
    await GoogleSignin.signOut();
  } catch {
    // Ignore sign-out errors
  }
}

/**
 * Revoke access and sign out
 *
 * This removes the app's access to the user's Google account.
 */
export async function revokeGoogleAccess(): Promise<void> {
  try {
    await GoogleSignin.revokeAccess();
    await GoogleSignin.signOut();
  } catch {
    // Ignore revoke errors
  }
}

/**
 * Get current access token
 *
 * Returns the current access token if signed in.
 */
export async function getGoogleTokens(): Promise<{
  accessToken: string;
  idToken: string | null;
} | null> {
  try {
    const tokens = await GoogleSignin.getTokens();
    return {
      accessToken: tokens.accessToken,
      idToken: tokens.idToken,
    };
  } catch {
    return null;
  }
}

/**
 * Check if user is currently signed in
 */
export async function isGoogleSignedIn(): Promise<boolean> {
  try {
    return await GoogleSignin.hasPreviousSignIn();
  } catch {
    return false;
  }
}

/**
 * Get current signed-in user
 */
export function getCurrentGoogleUser(): GoogleUser | null {
  try {
    return GoogleSignin.getCurrentUser();
  } catch {
    return null;
  }
}

/**
 * Handle sign-in errors and convert to GoogleSignInError
 */
function handleSignInError(error: unknown): GoogleSignInError {
  if (isGoogleSignInError(error)) {
    switch (error.code) {
      case statusCodes.SIGN_IN_CANCELLED:
        return {
          success: false,
          error: 'cancelled',
          message: 'Sign-in was cancelled',
        };
      case statusCodes.IN_PROGRESS:
        return {
          success: false,
          error: 'in_progress',
          message: 'Sign-in is already in progress',
        };
      case statusCodes.PLAY_SERVICES_NOT_AVAILABLE:
        return {
          success: false,
          error: 'play_services_unavailable',
          message: 'Google Play Services not available',
        };
      default:
        return {
          success: false,
          error: 'unknown',
          message: error.message || 'Unknown error during sign-in',
        };
    }
  }

  return {
    success: false,
    error: 'unknown',
    message: error instanceof Error ? error.message : 'Unknown error',
  };
}

/**
 * Type guard for Google Sign-In errors
 */
function isGoogleSignInError(error: unknown): error is { code: string; message?: string } {
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    typeof (error as { code: unknown }).code === 'string'
  );
}
