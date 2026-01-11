import * as SecureStore from 'expo-secure-store';

/**
 * Key used to store auth token in SecureStore
 */
const TOKEN_KEY = 'clerk-token';

/**
 * Token cache interface for Clerk
 */
export interface TokenCacheInterface {
  getToken: (key: string) => Promise<string | null>;
  saveToken: (key: string, value: string) => Promise<void>;
  clearToken: (key: string) => Promise<void>;
}

/**
 * TokenCache implementation using expo-secure-store
 *
 * This is used by ClerkProvider to persist auth tokens securely
 * across app restarts.
 */
export const tokenCache: TokenCacheInterface = {
  async getToken(key: string): Promise<string | null> {
    try {
      const value = await SecureStore.getItemAsync(`${TOKEN_KEY}-${key}`);
      return value;
    } catch {
      // Failed to get token, return null
      return null;
    }
  },

  async saveToken(key: string, value: string): Promise<void> {
    try {
      await SecureStore.setItemAsync(`${TOKEN_KEY}-${key}`, value);
    } catch {
      // Failed to save token, silently ignore
    }
  },

  async clearToken(key: string): Promise<void> {
    try {
      await SecureStore.deleteItemAsync(`${TOKEN_KEY}-${key}`);
    } catch {
      // Failed to delete token, silently ignore
    }
  },
};
