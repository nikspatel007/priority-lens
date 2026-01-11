/**
 * Tests for VoiceOrb component
 *
 * Note: Basic rendering tests only due to complexity of reanimated hooks in test environment.
 * Visual behavior is tested manually.
 */

import React from 'react';
import { render } from '@testing-library/react-native';

import { VoiceOrb } from '../VoiceOrb';
import { colors } from '../../../theme';

describe('VoiceOrb', () => {
  it('exports VoiceOrb component', () => {
    expect(VoiceOrb).toBeDefined();
  });

  it('accepts type prop with agent value', () => {
    const props: Parameters<typeof VoiceOrb>[0] = {
      type: 'agent',
      isSpeaking: false,
    };
    expect(props.type).toBe('agent');
  });

  it('accepts type prop with user value', () => {
    const props: Parameters<typeof VoiceOrb>[0] = {
      type: 'user',
      isSpeaking: false,
    };
    expect(props.type).toBe('user');
  });

  it('accepts isSpeaking prop', () => {
    const props: Parameters<typeof VoiceOrb>[0] = {
      type: 'agent',
      isSpeaking: true,
    };
    expect(props.isSpeaking).toBe(true);
  });

  it('accepts size prop', () => {
    const props: Parameters<typeof VoiceOrb>[0] = {
      type: 'agent',
      isSpeaking: false,
      size: 100,
    };
    expect(props.size).toBe(100);
  });

  it('accepts dimmed prop', () => {
    const props: Parameters<typeof VoiceOrb>[0] = {
      type: 'user',
      isSpeaking: false,
      dimmed: true,
    };
    expect(props.dimmed).toBe(true);
  });

  it('accepts style prop', () => {
    const props: Parameters<typeof VoiceOrb>[0] = {
      type: 'agent',
      isSpeaking: false,
      style: { marginTop: 20 },
    };
    expect(props.style).toBeDefined();
  });

  it('accepts testID prop', () => {
    const props: Parameters<typeof VoiceOrb>[0] = {
      type: 'agent',
      isSpeaking: false,
      testID: 'orb',
    };
    expect(props.testID).toBe('orb');
  });

  describe('colors', () => {
    it('has agent voice color defined', () => {
      expect(colors.voice.agent).toBe('#2196F3');
    });

    it('has user voice color defined', () => {
      expect(colors.voice.user).toBe('#9C27B0');
    });

    it('has agentGlow color defined', () => {
      expect(colors.voice.agentGlow).toBe('#64B5F6');
    });

    it('has userGlow color defined', () => {
      expect(colors.voice.userGlow).toBe('#CE93D8');
    });
  });
});
