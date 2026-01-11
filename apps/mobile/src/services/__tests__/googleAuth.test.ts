import {
  GoogleSignin,
  statusCodes,
} from '@react-native-google-signin/google-signin';
import {
  configureGoogleSignIn,
  signInWithGoogle,
  silentSignIn,
  signOutGoogle,
  revokeGoogleAccess,
  getGoogleTokens,
  isGoogleSignedIn,
  getCurrentGoogleUser,
  GOOGLE_SCOPES,
} from '../googleAuth';

// Mock Google Sign-In
jest.mock('@react-native-google-signin/google-signin', () => ({
  GoogleSignin: {
    configure: jest.fn(),
    hasPlayServices: jest.fn(),
    signIn: jest.fn(),
    signInSilently: jest.fn(),
    signOut: jest.fn(),
    revokeAccess: jest.fn(),
    getTokens: jest.fn(),
    hasPreviousSignIn: jest.fn(),
    getCurrentUser: jest.fn(),
  },
  statusCodes: {
    SIGN_IN_CANCELLED: 'SIGN_IN_CANCELLED',
    IN_PROGRESS: 'IN_PROGRESS',
    PLAY_SERVICES_NOT_AVAILABLE: 'PLAY_SERVICES_NOT_AVAILABLE',
  },
}));

const mockGoogleSignin = GoogleSignin as jest.Mocked<typeof GoogleSignin>;

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

describe('googleAuth', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('GOOGLE_SCOPES', () => {
    it('includes gmail.readonly scope', () => {
      expect(GOOGLE_SCOPES).toContain('https://www.googleapis.com/auth/gmail.readonly');
    });

    it('includes calendar.readonly scope', () => {
      expect(GOOGLE_SCOPES).toContain('https://www.googleapis.com/auth/calendar.readonly');
    });
  });

  describe('configureGoogleSignIn', () => {
    it('configures with web client ID', () => {
      configureGoogleSignIn('web-client-id');

      expect(mockGoogleSignin.configure).toHaveBeenCalledWith({
        webClientId: 'web-client-id',
        iosClientId: undefined,
        scopes: [...GOOGLE_SCOPES],
        offlineAccess: true,
        forceCodeForRefreshToken: true,
      });
    });

    it('configures with both web and iOS client IDs', () => {
      configureGoogleSignIn('web-client-id', 'ios-client-id');

      expect(mockGoogleSignin.configure).toHaveBeenCalledWith({
        webClientId: 'web-client-id',
        iosClientId: 'ios-client-id',
        scopes: [...GOOGLE_SCOPES],
        offlineAccess: true,
        forceCodeForRefreshToken: true,
      });
    });
  });

  describe('signInWithGoogle', () => {
    it('returns success with user data', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockResolvedValue({
        type: 'success',
        data: mockUser,
      });

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });
    });

    it('returns success with null tokens when not provided', async () => {
      const userWithoutTokens = { ...mockUser, idToken: null, serverAuthCode: null };
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockResolvedValue({
        type: 'success',
        data: userWithoutTokens,
      });

      const result = await signInWithGoogle();

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.idToken).toBeNull();
        expect(result.serverAuthCode).toBeNull();
      }
    });

    it('handles cancelled sign-in', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockResolvedValue({
        type: 'cancelled',
        data: null,
      });

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'cancelled',
        message: 'Sign-in was cancelled by user',
      });
    });

    it('handles missing user data', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockResolvedValue({
        type: 'success',
        data: null,
      });

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'unknown',
        message: 'No user data returned from Google',
      });
    });

    it('handles SIGN_IN_CANCELLED error', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockRejectedValue({
        code: statusCodes.SIGN_IN_CANCELLED,
        message: 'Cancelled',
      });

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'cancelled',
        message: 'Sign-in was cancelled',
      });
    });

    it('handles IN_PROGRESS error', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockRejectedValue({
        code: statusCodes.IN_PROGRESS,
      });

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'in_progress',
        message: 'Sign-in is already in progress',
      });
    });

    it('handles PLAY_SERVICES_NOT_AVAILABLE error', async () => {
      mockGoogleSignin.hasPlayServices.mockRejectedValue({
        code: statusCodes.PLAY_SERVICES_NOT_AVAILABLE,
      });

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'play_services_unavailable',
        message: 'Google Play Services not available',
      });
    });

    it('handles unknown error with code', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockRejectedValue({
        code: 'UNKNOWN_ERROR',
        message: 'Something went wrong',
      });

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'unknown',
        message: 'Something went wrong',
      });
    });

    it('handles unknown error without message', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockRejectedValue({
        code: 'UNKNOWN_ERROR',
      });

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'unknown',
        message: 'Unknown error during sign-in',
      });
    });

    it('handles Error instance', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockRejectedValue(new Error('Network error'));

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'unknown',
        message: 'Network error',
      });
    });

    it('handles non-Error exception', async () => {
      mockGoogleSignin.hasPlayServices.mockResolvedValue(true);
      mockGoogleSignin.signIn.mockRejectedValue('String error');

      const result = await signInWithGoogle();

      expect(result).toEqual({
        success: false,
        error: 'unknown',
        message: 'Unknown error',
      });
    });
  });

  describe('silentSignIn', () => {
    it('returns success when previously signed in', async () => {
      mockGoogleSignin.hasPreviousSignIn.mockResolvedValue(true);
      mockGoogleSignin.signInSilently.mockResolvedValue({
        type: 'success',
        data: mockUser,
      });

      const result = await silentSignIn();

      expect(result).toEqual({
        success: true,
        user: mockUser,
        idToken: 'mock-id-token',
        serverAuthCode: 'mock-server-auth-code',
      });
    });

    it('returns success with null tokens when not provided', async () => {
      const userWithoutTokens = { ...mockUser, idToken: null, serverAuthCode: null };
      mockGoogleSignin.hasPreviousSignIn.mockResolvedValue(true);
      mockGoogleSignin.signInSilently.mockResolvedValue({
        type: 'success',
        data: userWithoutTokens,
      });

      const result = await silentSignIn();

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.idToken).toBeNull();
        expect(result.serverAuthCode).toBeNull();
      }
    });

    it('returns error when no previous session', async () => {
      mockGoogleSignin.hasPreviousSignIn.mockResolvedValue(false);

      const result = await silentSignIn();

      expect(result).toEqual({
        success: false,
        error: 'cancelled',
        message: 'No previous session found',
      });
    });

    it('handles noSavedCredentialFound', async () => {
      mockGoogleSignin.hasPreviousSignIn.mockResolvedValue(true);
      mockGoogleSignin.signInSilently.mockResolvedValue({
        type: 'noSavedCredentialFound',
        data: null,
      });

      const result = await silentSignIn();

      expect(result).toEqual({
        success: false,
        error: 'cancelled',
        message: 'No saved credentials found',
      });
    });

    it('handles missing user data from silent sign-in', async () => {
      mockGoogleSignin.hasPreviousSignIn.mockResolvedValue(true);
      mockGoogleSignin.signInSilently.mockResolvedValue({
        type: 'success',
        data: null,
      });

      const result = await silentSignIn();

      expect(result).toEqual({
        success: false,
        error: 'unknown',
        message: 'No user data returned from silent sign-in',
      });
    });

    it('handles errors during silent sign-in', async () => {
      mockGoogleSignin.hasPreviousSignIn.mockResolvedValue(true);
      mockGoogleSignin.signInSilently.mockRejectedValue({
        code: statusCodes.SIGN_IN_CANCELLED,
      });

      const result = await silentSignIn();

      expect(result).toEqual({
        success: false,
        error: 'cancelled',
        message: 'Sign-in was cancelled',
      });
    });
  });

  describe('signOutGoogle', () => {
    it('calls GoogleSignin.signOut', async () => {
      mockGoogleSignin.signOut.mockResolvedValue(null);

      await signOutGoogle();

      expect(mockGoogleSignin.signOut).toHaveBeenCalled();
    });

    it('ignores sign-out errors', async () => {
      mockGoogleSignin.signOut.mockRejectedValue(new Error('Sign out failed'));

      // Should not throw
      await expect(signOutGoogle()).resolves.toBeUndefined();
    });
  });

  describe('revokeGoogleAccess', () => {
    it('revokes access and signs out', async () => {
      mockGoogleSignin.revokeAccess.mockResolvedValue(null);
      mockGoogleSignin.signOut.mockResolvedValue(null);

      await revokeGoogleAccess();

      expect(mockGoogleSignin.revokeAccess).toHaveBeenCalled();
      expect(mockGoogleSignin.signOut).toHaveBeenCalled();
    });

    it('ignores revoke errors', async () => {
      mockGoogleSignin.revokeAccess.mockRejectedValue(new Error('Revoke failed'));

      // Should not throw
      await expect(revokeGoogleAccess()).resolves.toBeUndefined();
    });
  });

  describe('getGoogleTokens', () => {
    it('returns tokens when signed in', async () => {
      mockGoogleSignin.getTokens.mockResolvedValue({
        accessToken: 'access-token-123',
        idToken: 'id-token-123',
      });

      const result = await getGoogleTokens();

      expect(result).toEqual({
        accessToken: 'access-token-123',
        idToken: 'id-token-123',
      });
    });

    it('returns null on error', async () => {
      mockGoogleSignin.getTokens.mockRejectedValue(new Error('Not signed in'));

      const result = await getGoogleTokens();

      expect(result).toBeNull();
    });
  });

  describe('isGoogleSignedIn', () => {
    it('returns true when signed in', async () => {
      mockGoogleSignin.hasPreviousSignIn.mockResolvedValue(true);

      const result = await isGoogleSignedIn();

      expect(result).toBe(true);
    });

    it('returns false when not signed in', async () => {
      mockGoogleSignin.hasPreviousSignIn.mockResolvedValue(false);

      const result = await isGoogleSignedIn();

      expect(result).toBe(false);
    });

    it('returns false on error', async () => {
      mockGoogleSignin.hasPreviousSignIn.mockRejectedValue(new Error('Error'));

      const result = await isGoogleSignedIn();

      expect(result).toBe(false);
    });
  });

  describe('getCurrentGoogleUser', () => {
    it('returns current user', () => {
      mockGoogleSignin.getCurrentUser.mockReturnValue(mockUser);

      const result = getCurrentGoogleUser();

      expect(result).toEqual(mockUser);
    });

    it('returns null when no user', () => {
      mockGoogleSignin.getCurrentUser.mockReturnValue(null);

      const result = getCurrentGoogleUser();

      expect(result).toBeNull();
    });

    it('returns null on error', () => {
      mockGoogleSignin.getCurrentUser.mockImplementation(() => {
        throw new Error('Error');
      });

      const result = getCurrentGoogleUser();

      expect(result).toBeNull();
    });
  });
});
