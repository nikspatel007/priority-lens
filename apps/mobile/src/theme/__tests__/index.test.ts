import {
  colors,
  typography,
  spacing,
  borderRadius,
  shadows,
  animation,
  theme,
} from '../index';

describe('Theme', () => {
  describe('colors', () => {
    it('has primary palette with all shades', () => {
      expect(colors.primary[50]).toBe('#E3F2FD');
      expect(colors.primary[500]).toBe('#2196F3');
      expect(colors.primary[900]).toBe('#0D47A1');
    });

    it('has gray palette with all shades', () => {
      expect(colors.gray[50]).toBe('#FAFAFA');
      expect(colors.gray[500]).toBe('#9E9E9E');
      expect(colors.gray[900]).toBe('#212121');
    });

    it('has semantic colors', () => {
      expect(colors.success).toBe('#4CAF50');
      expect(colors.warning).toBe('#FF9800');
      expect(colors.error).toBe('#F44336');
      expect(colors.info).toBe('#2196F3');
    });

    it('has background colors', () => {
      expect(colors.background).toBe('#FFFFFF');
      expect(colors.backgrounds.primary).toBe('#FFFFFF');
      expect(colors.backgrounds.secondary).toBe('#F5F5F5');
      expect(colors.backgrounds.tertiary).toBe('#EEEEEE');
    });

    it('has text colors', () => {
      expect(colors.text.primary).toBe('#212121');
      expect(colors.text.secondary).toBe('#757575');
      expect(colors.text.disabled).toBe('#9E9E9E');
      expect(colors.text.inverse).toBe('#FFFFFF');
    });

    it('has surface colors', () => {
      expect(colors.surface).toBe('#FFFFFF');
      expect(colors.surfaces.card).toBe('#FFFFFF');
      expect(colors.surfaces.elevated).toBe('#FFFFFF');
    });

    it('has border colors', () => {
      expect(colors.border.light).toBe('#E0E0E0');
      expect(colors.border.medium).toBe('#BDBDBD');
      expect(colors.border.dark).toBe('#9E9E9E');
    });

    it('has voice UI colors', () => {
      expect(colors.voice.agent).toBe('#2196F3');
      expect(colors.voice.agentGlow).toBe('#64B5F6');
      expect(colors.voice.user).toBe('#9C27B0');
      expect(colors.voice.userGlow).toBe('#CE93D8');
      expect(colors.voice.waveform).toBe('#BDBDBD');
      expect(colors.voice.waveformActive).toBe('#2196F3');
    });
  });

  describe('typography', () => {
    it('has font families', () => {
      expect(typography.fontFamily.serif).toBe('Georgia');
      expect(typography.fontFamily.sans).toBe('System');
      expect(typography.fontFamily.mono).toBe('Menlo');
    });

    it('has font sizes following scale', () => {
      expect(typography.fontSize.xs).toBe(12);
      expect(typography.fontSize.base).toBe(16);
      expect(typography.fontSize['4xl']).toBe(36);
    });

    it('has font weights', () => {
      expect(typography.fontWeight.normal).toBe('400');
      expect(typography.fontWeight.bold).toBe('700');
    });

    it('has line heights', () => {
      expect(typography.lineHeight.tight).toBe(1.25);
      expect(typography.lineHeight.normal).toBe(1.5);
      expect(typography.lineHeight.relaxed).toBe(1.75);
    });
  });

  describe('spacing', () => {
    it('uses 8px base grid', () => {
      expect(spacing[0]).toBe(0);
      expect(spacing[1]).toBe(8);
      expect(spacing[2]).toBe(16);
      expect(spacing[4]).toBe(32);
    });

    it('has half-step values', () => {
      expect(spacing[0.5]).toBe(4);
      expect(spacing[1.5]).toBe(12);
      expect(spacing[2.5]).toBe(20);
    });

    it('has larger values', () => {
      expect(spacing[8]).toBe(64);
      expect(spacing[12]).toBe(96);
      expect(spacing[16]).toBe(128);
    });
  });

  describe('borderRadius', () => {
    it('has radius scale', () => {
      expect(borderRadius.none).toBe(0);
      expect(borderRadius.sm).toBe(4);
      expect(borderRadius.md).toBe(8);
      expect(borderRadius.lg).toBe(12);
      expect(borderRadius.xl).toBe(16);
      expect(borderRadius['2xl']).toBe(24);
      expect(borderRadius.full).toBe(9999);
    });
  });

  describe('shadows', () => {
    it('has shadow presets', () => {
      expect(shadows.none.elevation).toBe(0);
      expect(shadows.sm.elevation).toBe(1);
      expect(shadows.md.elevation).toBe(3);
      expect(shadows.lg.elevation).toBe(6);
      expect(shadows.xl.elevation).toBe(12);
    });

    it('shadows have proper structure', () => {
      expect(shadows.md).toEqual({
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 3,
      });
    });
  });

  describe('animation', () => {
    it('has duration presets', () => {
      expect(animation.duration.fast).toBe(150);
      expect(animation.duration.normal).toBe(300);
      expect(animation.duration.slow).toBe(500);
    });

    it('has easing presets', () => {
      expect(animation.easing.easeInOut).toBe('ease-in-out');
      expect(animation.easing.easeIn).toBe('ease-in');
      expect(animation.easing.easeOut).toBe('ease-out');
    });
  });

  describe('theme object', () => {
    it('includes base color palettes', () => {
      // Base palettes are spread into theme.colors
      expect(theme.colors.gray[500]).toBe(colors.gray[500]);
      expect(theme.colors.success).toBe(colors.success);
    });

    it('includes color aliases', () => {
      expect(theme.colors.primary).toBe('#2196F3');
      expect(theme.colors.primaryLight).toBe('#90CAF9');
      expect(theme.colors.secondary).toBe('#9C27B0');
      expect(theme.colors.secondaryLight).toBe('#E1BEE7');
      expect(theme.colors.surface).toBe('#FFFFFF');
      expect(theme.colors.textPrimary).toBe('#212121');
      expect(theme.colors.textSecondary).toBe('#757575');
      expect(theme.colors.white).toBe('#FFFFFF');
      expect(theme.colors.black).toBe('#000000');
      expect(theme.colors.disabled).toBe('#BDBDBD');
      expect(theme.colors.error).toBe('#F44336');
    });

    it('includes typography', () => {
      expect(theme.typography).toBe(typography);
    });

    it('includes spacing with aliases', () => {
      expect(theme.spacing[1]).toBe(8);
      expect(theme.spacing.xs).toBe(4);
      expect(theme.spacing.sm).toBe(8);
      expect(theme.spacing.md).toBe(16);
      expect(theme.spacing.lg).toBe(24);
      expect(theme.spacing.xl).toBe(32);
    });

    it('includes fontSizes', () => {
      expect(theme.fontSizes.xs).toBe(typography.fontSize.xs);
      expect(theme.fontSizes.sm).toBe(typography.fontSize.sm);
      expect(theme.fontSizes.base).toBe(typography.fontSize.base);
      expect(theme.fontSizes.md).toBe(typography.fontSize.base); // md is an alias for base
      expect(theme.fontSizes.lg).toBe(typography.fontSize.lg);
    });

    it('includes borderRadius', () => {
      expect(theme.borderRadius).toBe(borderRadius);
    });

    it('includes shadows', () => {
      expect(theme.shadows).toBe(shadows);
    });

    it('includes animation', () => {
      expect(theme.animation).toBe(animation);
    });
  });
});
