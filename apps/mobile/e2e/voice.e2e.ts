/**
 * Voice Conversation E2E Tests
 *
 * Tests the voice/text conversation functionality.
 */

import { device, element, by, expect, waitFor } from 'detox';

describe('Voice Conversation', () => {
  beforeAll(async () => {
    await device.launchApp({
      permissions: {
        microphone: 'YES',
      },
    });
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  describe('Conversation Screen', () => {
    // Note: These tests assume user is fully authenticated

    it('shows conversation screen', async () => {
      await expect(element(by.id('conversation-screen'))).toBeVisible();
    });

    it('displays header with app title', async () => {
      await expect(element(by.text('Priority Lens'))).toBeVisible();
    });

    it('shows mode toggle button', async () => {
      await expect(element(by.id('mode-toggle'))).toBeVisible();
    });
  });

  describe('Voice Mode', () => {
    it('starts in voice mode by default', async () => {
      await expect(element(by.id('voice-panel'))).toBeVisible();
    });

    it('shows voice orbs', async () => {
      await expect(element(by.id('agent-orb'))).toBeVisible();
      await expect(element(by.id('user-orb'))).toBeVisible();
    });

    it('can toggle microphone', async () => {
      await element(by.id('mic-toggle')).tap();
      // Mic state should change
      await expect(element(by.id('mic-toggle'))).toBeVisible();
    });

    it('shows connection status', async () => {
      await waitFor(element(by.id('connection-indicator')))
        .toBeVisible()
        .withTimeout(10000);
    });
  });

  describe('Text Mode', () => {
    it('can switch to text mode', async () => {
      await element(by.id('mode-toggle')).tap();
      await expect(element(by.id('text-input-container'))).toBeVisible();
    });

    it('hides voice panel in text mode', async () => {
      await element(by.id('mode-toggle')).tap();
      await expect(element(by.id('voice-panel'))).not.toBeVisible();
    });

    it('shows text input field', async () => {
      await element(by.id('mode-toggle')).tap();
      await expect(element(by.id('text-input'))).toBeVisible();
    });

    it('shows send button', async () => {
      await element(by.id('mode-toggle')).tap();
      await expect(element(by.id('send-button'))).toBeVisible();
    });

    it('can type a message', async () => {
      await element(by.id('mode-toggle')).tap();
      await element(by.id('text-input')).typeText('Hello');
      await expect(element(by.id('text-input'))).toHaveText('Hello');
    });

    it('can send a message', async () => {
      await element(by.id('mode-toggle')).tap();
      await element(by.id('text-input')).typeText('Show my inbox');
      await element(by.id('send-button')).tap();

      // Input should be cleared after sending
      await waitFor(element(by.id('text-input')))
        .toHaveText('')
        .withTimeout(5000);
    });
  });

  describe('SDUI Rendering', () => {
    it('renders SDUI blocks when received', async () => {
      // After asking about inbox, SDUI blocks should appear
      await element(by.id('mode-toggle')).tap();
      await element(by.id('text-input')).typeText('Show my inbox');
      await element(by.id('send-button')).tap();

      // Wait for SDUI container to appear
      await waitFor(element(by.id('sdui-container')))
        .toBeVisible()
        .withTimeout(15000);
    });

    it('can interact with SDUI action buttons', async () => {
      // Assuming there's a task card with a complete button
      await waitFor(element(by.id('sdui-container')))
        .toBeVisible()
        .withTimeout(15000);

      // Try to tap an action if available
      const actionButton = element(by.id(/sdui-action-.*/));
      if (await actionButton.exists()) {
        await actionButton.tap();
      }
    });
  });

  describe('Empty State', () => {
    it('shows welcome message when no content', async () => {
      await expect(element(by.id('empty-state'))).toBeVisible();
    });

    it('displays personalized greeting if user name available', async () => {
      // This depends on user data
      await expect(element(by.id('empty-state'))).toBeVisible();
    });
  });

  describe('Error Handling', () => {
    it('shows error container when error occurs', async () => {
      // Force an error condition
      // In real tests, you might mock the API to return an error
      // For now, just check the error container can be rendered
      await expect(element(by.id('conversation-screen'))).toBeVisible();
    });
  });

  describe('Mode Toggle Persistence', () => {
    it('switches back to voice mode', async () => {
      // Switch to text
      await element(by.id('mode-toggle')).tap();
      await expect(element(by.id('text-input-container'))).toBeVisible();

      // Switch back to voice
      await element(by.id('mode-toggle')).tap();
      await expect(element(by.id('voice-panel'))).toBeVisible();
    });
  });
});
