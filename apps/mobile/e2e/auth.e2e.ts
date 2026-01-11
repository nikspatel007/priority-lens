/**
 * Authentication Flow E2E Tests
 *
 * Tests the complete user authentication journey from sign-in to main app.
 */

import { device, element, by, expect, waitFor } from 'detox';

describe('Authentication Flow', () => {
  beforeAll(async () => {
    await device.launchApp();
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  describe('Sign In Screen', () => {
    it('shows sign in screen when not authenticated', async () => {
      await expect(element(by.id('signin-screen'))).toBeVisible();
    });

    it('displays Clerk sign-in component', async () => {
      await expect(element(by.id('clerk-signin'))).toBeVisible();
    });

    it('shows app branding', async () => {
      await expect(element(by.text('Priority Lens'))).toBeVisible();
    });
  });

  describe('Landing Screen', () => {
    // Note: These tests require being signed in to Clerk first
    // In a real scenario, you would set up auth state before these tests

    it('shows Google connect button after Clerk sign-in', async () => {
      // This test assumes user is signed in but Google not connected
      // In production, you'd use device.launchApp with a pre-authenticated state
      await expect(element(by.id('landing-screen'))).toBeVisible();
      await expect(element(by.id('connect-google-button'))).toBeVisible();
    });

    it('displays welcome message', async () => {
      await expect(element(by.text('Connect Your Google Account'))).toBeVisible();
    });
  });

  describe('Sync Progress Screen', () => {
    it('shows sync progress after Google connection', async () => {
      // This test assumes Google was just connected
      await expect(element(by.id('sync-progress-screen'))).toBeVisible();
    });

    it('displays progress indicator', async () => {
      await expect(element(by.id('sync-progress-indicator'))).toBeVisible();
    });

    it('shows status message', async () => {
      await expect(element(by.id('sync-status-text'))).toBeVisible();
    });
  });

  describe('Main App Navigation', () => {
    // These tests assume full authentication is complete

    it('shows conversation screen after sync complete', async () => {
      await expect(element(by.id('conversation-screen'))).toBeVisible();
    });

    it('can navigate to settings', async () => {
      await element(by.id('settings-button')).tap();
      await expect(element(by.id('settings-screen'))).toBeVisible();
    });

    it('can navigate back from settings', async () => {
      await element(by.id('settings-button')).tap();
      await expect(element(by.id('settings-screen'))).toBeVisible();
      await element(by.id('back-button')).tap();
      await expect(element(by.id('conversation-screen'))).toBeVisible();
    });
  });

  describe('Sign Out', () => {
    it('can sign out from settings', async () => {
      await element(by.id('settings-button')).tap();
      await element(by.id('sign-out-button')).tap();

      // Confirm sign out in alert
      await element(by.text('Sign Out')).atIndex(1).tap();

      // Should return to sign in screen
      await waitFor(element(by.id('signin-screen')))
        .toBeVisible()
        .withTimeout(5000);
    });
  });
});
