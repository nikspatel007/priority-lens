/**
 * Tests for voice components index
 */

import {
  VoiceOrb,
  Waveform,
  VoiceModePanel,
  type VoiceOrbProps,
  type WaveformProps,
  type VoiceModePanelProps,
  type OrbType,
  type SpeakingSource,
} from '../index';

describe('Voice Components Index', () => {
  it('exports VoiceOrb', () => {
    expect(VoiceOrb).toBeDefined();
  });

  it('exports Waveform', () => {
    expect(Waveform).toBeDefined();
  });

  it('exports VoiceModePanel', () => {
    expect(VoiceModePanel).toBeDefined();
  });

  // Type tests (these validate at compile time)
  it('exports VoiceOrbProps type', () => {
    const props: VoiceOrbProps = {
      type: 'agent',
      isSpeaking: false,
    };
    expect(props.type).toBe('agent');
  });

  it('exports WaveformProps type', () => {
    const props: WaveformProps = {
      speakingSource: 'none',
    };
    expect(props.speakingSource).toBe('none');
  });

  it('exports VoiceModePanelProps type', () => {
    const props: VoiceModePanelProps = {
      speakingSource: 'none',
      isMicrophoneEnabled: false,
      isConnected: true,
      isConnecting: false,
      onToggleMicrophone: () => {},
      onToggleMode: () => {},
    };
    expect(props.speakingSource).toBe('none');
  });

  it('exports OrbType type', () => {
    const orbType: OrbType = 'user';
    expect(orbType).toBe('user');
  });

  it('exports SpeakingSource type', () => {
    const source: SpeakingSource = 'agent';
    expect(source).toBe('agent');
  });
});
