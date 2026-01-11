import * as SecureStore from 'expo-secure-store';
import { tokenCache } from '../TokenCache';

// Mock expo-secure-store
jest.mock('expo-secure-store', () => ({
  getItemAsync: jest.fn(),
  setItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));

const mockSecureStore = SecureStore as jest.Mocked<typeof SecureStore>;

describe('TokenCache', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getToken', () => {
    it('returns stored token value', async () => {
      mockSecureStore.getItemAsync.mockResolvedValue('stored-token-value');

      const result = await tokenCache.getToken('session');

      expect(mockSecureStore.getItemAsync).toHaveBeenCalledWith('clerk-token-session');
      expect(result).toBe('stored-token-value');
    });

    it('returns null when no token exists', async () => {
      mockSecureStore.getItemAsync.mockResolvedValue(null);

      const result = await tokenCache.getToken('session');

      expect(result).toBeNull();
    });

    it('returns null on error', async () => {
      mockSecureStore.getItemAsync.mockRejectedValue(new Error('Storage error'));

      const result = await tokenCache.getToken('session');

      expect(result).toBeNull();
    });
  });

  describe('saveToken', () => {
    it('saves token to secure store', async () => {
      mockSecureStore.setItemAsync.mockResolvedValue();

      await tokenCache.saveToken('session', 'new-token-value');

      expect(mockSecureStore.setItemAsync).toHaveBeenCalledWith(
        'clerk-token-session',
        'new-token-value'
      );
    });

    it('handles save error silently', async () => {
      mockSecureStore.setItemAsync.mockRejectedValue(new Error('Save failed'));

      // Should not throw
      await expect(tokenCache.saveToken('session', 'token')).resolves.toBeUndefined();
    });
  });

  describe('clearToken', () => {
    it('deletes token from secure store', async () => {
      mockSecureStore.deleteItemAsync.mockResolvedValue();

      await tokenCache.clearToken('session');

      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith('clerk-token-session');
    });

    it('handles delete error silently', async () => {
      mockSecureStore.deleteItemAsync.mockRejectedValue(new Error('Delete failed'));

      // Should not throw
      await expect(tokenCache.clearToken('session')).resolves.toBeUndefined();
    });
  });

  describe('key prefixing', () => {
    it('prefixes all keys with clerk-token-', async () => {
      mockSecureStore.getItemAsync.mockResolvedValue('value');
      mockSecureStore.setItemAsync.mockResolvedValue();
      mockSecureStore.deleteItemAsync.mockResolvedValue();

      await tokenCache.getToken('my-key');
      await tokenCache.saveToken('my-key', 'value');
      await tokenCache.clearToken('my-key');

      expect(mockSecureStore.getItemAsync).toHaveBeenCalledWith('clerk-token-my-key');
      expect(mockSecureStore.setItemAsync).toHaveBeenCalledWith('clerk-token-my-key', 'value');
      expect(mockSecureStore.deleteItemAsync).toHaveBeenCalledWith('clerk-token-my-key');
    });
  });
});
