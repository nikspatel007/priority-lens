/**
 * SDUI Primitives Tests
 *
 * Tests for primitive SDUI components.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { SDUIText } from '../primitives/SDUIText';
import { SDUIButton } from '../primitives/SDUIButton';
import { SDUIBadge } from '../primitives/SDUIBadge';
import { SDUIAvatar } from '../primitives/SDUIAvatar';
import { SDUIGradientAvatar } from '../primitives/SDUIGradientAvatar';
import { SDUIViewToggle, ViewMode } from '../primitives/SDUIViewToggle';

describe('SDUIText', () => {
  it('renders text value', () => {
    const { getByText } = render(<SDUIText value="Hello World" />);
    expect(getByText('Hello World')).toBeTruthy();
  });

  it('renders with body variant by default', () => {
    const { getByText } = render(<SDUIText value="Body Text" />);
    expect(getByText('Body Text')).toBeTruthy();
  });

  it('renders with heading variant', () => {
    const { getByText } = render(<SDUIText value="Heading" variant="heading" />);
    expect(getByText('Heading')).toBeTruthy();
  });

  it('renders with title variant', () => {
    const { getByText } = render(<SDUIText value="Title" variant="title" />);
    expect(getByText('Title')).toBeTruthy();
  });

  it('renders with caption variant', () => {
    const { getByText } = render(<SDUIText value="Caption" variant="caption" />);
    expect(getByText('Caption')).toBeTruthy();
  });

  it('renders with label variant', () => {
    const { getByText } = render(<SDUIText value="Label" variant="label" />);
    expect(getByText('Label')).toBeTruthy();
  });

  it('applies custom color', () => {
    const { getByText } = render(<SDUIText value="Colored" color="#FF0000" />);
    expect(getByText('Colored')).toBeTruthy();
  });

  it('applies weight', () => {
    const { getByText } = render(<SDUIText value="Bold" weight="bold" />);
    expect(getByText('Bold')).toBeTruthy();
  });

  it('applies all weight variants', () => {
    const { getByText, rerender } = render(
      <SDUIText value="Normal" weight="normal" />
    );
    expect(getByText('Normal')).toBeTruthy();

    rerender(<SDUIText value="Medium" weight="medium" />);
    expect(getByText('Medium')).toBeTruthy();

    rerender(<SDUIText value="Semibold" weight="semibold" />);
    expect(getByText('Semibold')).toBeTruthy();
  });

  it('applies text alignment', () => {
    const { getByText, rerender } = render(
      <SDUIText value="Left" align="left" />
    );
    expect(getByText('Left')).toBeTruthy();

    rerender(<SDUIText value="Center" align="center" />);
    expect(getByText('Center')).toBeTruthy();

    rerender(<SDUIText value="Right" align="right" />);
    expect(getByText('Right')).toBeTruthy();
  });

  it('limits number of lines', () => {
    const { getByText } = render(
      <SDUIText value="Long text here" numberOfLines={1} />
    );
    expect(getByText('Long text here')).toBeTruthy();
  });
});

describe('SDUIButton', () => {
  it('renders button with label', () => {
    const { getByText } = render(<SDUIButton label="Click Me" />);
    expect(getByText('Click Me')).toBeTruthy();
  });

  it('calls onPress when pressed', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <SDUIButton label="Press Me" onPress={onPress} />
    );
    fireEvent.press(getByText('Press Me'));
    expect(onPress).toHaveBeenCalled();
  });

  it('renders with primary variant by default', () => {
    const { getByTestId } = render(<SDUIButton label="Primary" />);
    expect(getByTestId('sdui-button')).toBeTruthy();
  });

  it('renders with secondary variant', () => {
    const { getByText } = render(
      <SDUIButton label="Secondary" variant="secondary" />
    );
    expect(getByText('Secondary')).toBeTruthy();
  });

  it('renders with outline variant', () => {
    const { getByText } = render(
      <SDUIButton label="Outline" variant="outline" />
    );
    expect(getByText('Outline')).toBeTruthy();
  });

  it('renders with ghost variant', () => {
    const { getByText } = render(<SDUIButton label="Ghost" variant="ghost" />);
    expect(getByText('Ghost')).toBeTruthy();
  });

  it('renders with destructive variant', () => {
    const { getByText } = render(
      <SDUIButton label="Destructive" variant="destructive" />
    );
    expect(getByText('Destructive')).toBeTruthy();
  });

  it('renders with small size', () => {
    const { getByText } = render(<SDUIButton label="Small" size="sm" />);
    expect(getByText('Small')).toBeTruthy();
  });

  it('renders with medium size by default', () => {
    const { getByText } = render(<SDUIButton label="Medium" />);
    expect(getByText('Medium')).toBeTruthy();
  });

  it('renders with large size', () => {
    const { getByText } = render(<SDUIButton label="Large" size="lg" />);
    expect(getByText('Large')).toBeTruthy();
  });

  it('shows loading indicator when loading', () => {
    const { queryByText, getByTestId } = render(
      <SDUIButton label="Loading" loading />
    );
    expect(queryByText('Loading')).toBeNull();
    expect(getByTestId('sdui-button')).toBeTruthy();
  });

  it('is disabled when loading', () => {
    const onPress = jest.fn();
    const { getByTestId } = render(
      <SDUIButton label="Loading" loading onPress={onPress} />
    );
    fireEvent.press(getByTestId('sdui-button'));
    expect(onPress).not.toHaveBeenCalled();
  });

  it('is disabled when disabled prop is true', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <SDUIButton label="Disabled" disabled onPress={onPress} />
    );
    fireEvent.press(getByText('Disabled'));
    expect(onPress).not.toHaveBeenCalled();
  });
});

describe('SDUIBadge', () => {
  it('renders badge with value', () => {
    const { getByText } = render(<SDUIBadge value="New" />);
    expect(getByText('New')).toBeTruthy();
  });

  it('renders with numeric value', () => {
    const { getByText } = render(<SDUIBadge value={42} />);
    expect(getByText('42')).toBeTruthy();
  });

  it('renders with default variant', () => {
    const { getByTestId } = render(<SDUIBadge value="Default" />);
    expect(getByTestId('sdui-badge')).toBeTruthy();
  });

  it('renders with success variant', () => {
    const { getByText } = render(<SDUIBadge value="Success" variant="success" />);
    expect(getByText('Success')).toBeTruthy();
  });

  it('renders with warning variant', () => {
    const { getByText } = render(<SDUIBadge value="Warning" variant="warning" />);
    expect(getByText('Warning')).toBeTruthy();
  });

  it('renders with error variant', () => {
    const { getByText } = render(<SDUIBadge value="Error" variant="error" />);
    expect(getByText('Error')).toBeTruthy();
  });

  it('renders with info variant', () => {
    const { getByText } = render(<SDUIBadge value="Info" variant="info" />);
    expect(getByText('Info')).toBeTruthy();
  });
});

describe('SDUIAvatar', () => {
  it('renders with image when src provided', () => {
    const { getByTestId } = render(
      <SDUIAvatar name="John Doe" src="https://example.com/avatar.jpg" />
    );
    expect(getByTestId('sdui-avatar-image')).toBeTruthy();
  });

  it('renders fallback with initials when no src', () => {
    const { getByTestId, getByText } = render(<SDUIAvatar name="John Doe" />);
    expect(getByTestId('sdui-avatar-fallback')).toBeTruthy();
    expect(getByText('JD')).toBeTruthy();
  });

  it('generates initials from single word name', () => {
    const { getByText } = render(<SDUIAvatar name="John" />);
    expect(getByText('JO')).toBeTruthy();
  });

  it('uses custom fallback text', () => {
    const { getByText } = render(
      <SDUIAvatar name="John Doe" fallback="AB" />
    );
    expect(getByText('AB')).toBeTruthy();
  });

  it('uses custom size', () => {
    const { getByTestId } = render(<SDUIAvatar name="John Doe" size={60} />);
    expect(getByTestId('sdui-avatar-fallback')).toBeTruthy();
  });

  it('uses default size of 40', () => {
    const { getByTestId } = render(<SDUIAvatar name="John" />);
    expect(getByTestId('sdui-avatar-fallback')).toBeTruthy();
  });

  it('generates consistent color from name', () => {
    const { getByTestId: get1 } = render(<SDUIAvatar name="Test Name" />);
    const { getByTestId: get2 } = render(<SDUIAvatar name="Test Name" />);
    expect(get1('sdui-avatar-fallback')).toBeTruthy();
    expect(get2('sdui-avatar-fallback')).toBeTruthy();
  });
});

describe('SDUIGradientAvatar', () => {
  it('renders with image when imageUrl provided', () => {
    const { getByTestId } = render(
      <SDUIGradientAvatar initials="JD" imageUrl="https://example.com/avatar.jpg" />
    );
    expect(getByTestId('sdui-gradient-avatar-image')).toBeTruthy();
  });

  it('renders with initials when no imageUrl', () => {
    const { getByTestId, getByText } = render(
      <SDUIGradientAvatar initials="JD" />
    );
    expect(getByTestId('sdui-gradient-avatar')).toBeTruthy();
    expect(getByText('JD')).toBeTruthy();
  });

  it('uses default initials when not provided', () => {
    const { getByText } = render(<SDUIGradientAvatar />);
    expect(getByText('?')).toBeTruthy();
  });

  it('uses gradient colors when provided', () => {
    const { getByTestId } = render(
      <SDUIGradientAvatar initials="AB" gradientColors={['#FF0000', '#00FF00']} />
    );
    expect(getByTestId('sdui-gradient-avatar')).toBeTruthy();
  });

  it('uses custom size', () => {
    const { getByTestId } = render(
      <SDUIGradientAvatar initials="XY" size={80} />
    );
    expect(getByTestId('sdui-gradient-avatar')).toBeTruthy();
  });

  it('uses default size of 56', () => {
    const { getByTestId } = render(<SDUIGradientAvatar initials="ZZ" />);
    expect(getByTestId('sdui-gradient-avatar')).toBeTruthy();
  });

  it('applies custom style', () => {
    const { getByTestId } = render(
      <SDUIGradientAvatar initials="ST" style={{ opacity: 0.5 }} />
    );
    expect(getByTestId('sdui-gradient-avatar')).toBeTruthy();
  });
});

describe('SDUIViewToggle', () => {
  it('renders view toggle', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(
      <SDUIViewToggle mode="cards" onChange={onChange} />
    );
    expect(getByTestId('sdui-view-toggle')).toBeTruthy();
  });

  it('shows cards option as active when mode is cards', () => {
    const onChange = jest.fn();
    const { getByText } = render(
      <SDUIViewToggle mode="cards" onChange={onChange} />
    );
    expect(getByText('Cards')).toBeTruthy();
    expect(getByText('List')).toBeTruthy();
  });

  it('shows list option as active when mode is list', () => {
    const onChange = jest.fn();
    const { getByText } = render(
      <SDUIViewToggle mode="list" onChange={onChange} />
    );
    expect(getByText('Cards')).toBeTruthy();
    expect(getByText('List')).toBeTruthy();
  });

  it('calls onChange with cards when cards pressed', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(
      <SDUIViewToggle mode="list" onChange={onChange} />
    );
    fireEvent.press(getByTestId('sdui-view-toggle-cards'));
    expect(onChange).toHaveBeenCalledWith('cards');
  });

  it('calls onChange with list when list pressed', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(
      <SDUIViewToggle mode="cards" onChange={onChange} />
    );
    fireEvent.press(getByTestId('sdui-view-toggle-list'));
    expect(onChange).toHaveBeenCalledWith('list');
  });

  it('can toggle between modes', () => {
    const onChange = jest.fn();
    const { getByTestId, rerender } = render(
      <SDUIViewToggle mode="cards" onChange={onChange} />
    );

    fireEvent.press(getByTestId('sdui-view-toggle-list'));
    expect(onChange).toHaveBeenCalledWith('list');

    rerender(<SDUIViewToggle mode="list" onChange={onChange} />);
    fireEvent.press(getByTestId('sdui-view-toggle-cards'));
    expect(onChange).toHaveBeenCalledWith('cards');
  });
});
