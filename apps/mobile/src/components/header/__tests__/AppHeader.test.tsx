/**
 * Tests for AppHeader component
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';

import { AppHeader } from '../AppHeader';

describe('AppHeader', () => {
  it('renders app title', () => {
    const { getByText } = render(<AppHeader testID="header" />);
    expect(getByText('Priority Lens')).toBeTruthy();
  });

  it('renders with testID', () => {
    const { getByTestId } = render(<AppHeader testID="header" />);
    expect(getByTestId('header')).toBeTruthy();
  });

  it('renders title with testID', () => {
    const { getByTestId } = render(<AppHeader testID="header" />);
    expect(getByTestId('header-title')).toBeTruthy();
  });

  it('applies top inset', () => {
    const { getByTestId } = render(<AppHeader testID="header" topInset={44} />);
    const container = getByTestId('header');
    expect(container.props.style).toEqual(
      expect.arrayContaining([expect.objectContaining({ paddingTop: 44 })])
    );
  });

  it('applies custom style', () => {
    const customStyle = { marginBottom: 10 };
    const { getByTestId } = render(<AppHeader testID="header" style={customStyle} />);
    const container = getByTestId('header');
    expect(container.props.style).toEqual(
      expect.arrayContaining([expect.objectContaining(customStyle)])
    );
  });

  describe('profile button', () => {
    it('renders profile button when onProfilePress is provided', () => {
      const onProfilePress = jest.fn();
      const { getByTestId } = render(
        <AppHeader testID="header" onProfilePress={onProfilePress} />
      );
      expect(getByTestId('header-profile')).toBeTruthy();
    });

    it('does not render profile button when onProfilePress is not provided', () => {
      const { queryByTestId } = render(<AppHeader testID="header" />);
      expect(queryByTestId('header-profile')).toBeNull();
    });

    it('calls onProfilePress when profile button is pressed', () => {
      const onProfilePress = jest.fn();
      const { getByTestId } = render(
        <AppHeader testID="header" onProfilePress={onProfilePress} />
      );
      fireEvent.press(getByTestId('header-profile'));
      expect(onProfilePress).toHaveBeenCalledTimes(1);
    });
  });

  describe('view toggle', () => {
    it('does not render view toggle by default', () => {
      const { queryByTestId } = render(<AppHeader testID="header" />);
      expect(queryByTestId('header-view-toggle')).toBeNull();
    });

    it('does not render view toggle when showViewToggle is false', () => {
      const { queryByTestId } = render(
        <AppHeader testID="header" showViewToggle={false} onViewModeChange={jest.fn()} />
      );
      expect(queryByTestId('header-view-toggle')).toBeNull();
    });

    it('renders view toggle when showViewToggle is true and onViewModeChange is provided', () => {
      const { getByTestId } = render(
        <AppHeader testID="header" showViewToggle={true} onViewModeChange={jest.fn()} />
      );
      expect(getByTestId('header-view-toggle')).toBeTruthy();
    });

    it('renders cards and list buttons', () => {
      const { getByTestId } = render(
        <AppHeader testID="header" showViewToggle={true} onViewModeChange={jest.fn()} />
      );
      expect(getByTestId('header-cards-btn')).toBeTruthy();
      expect(getByTestId('header-list-btn')).toBeTruthy();
    });

    it('calls onViewModeChange with cards when cards button is pressed', () => {
      const onViewModeChange = jest.fn();
      const { getByTestId } = render(
        <AppHeader
          testID="header"
          showViewToggle={true}
          onViewModeChange={onViewModeChange}
          viewMode="list"
        />
      );
      fireEvent.press(getByTestId('header-cards-btn'));
      expect(onViewModeChange).toHaveBeenCalledWith('cards');
    });

    it('calls onViewModeChange with list when list button is pressed', () => {
      const onViewModeChange = jest.fn();
      const { getByTestId } = render(
        <AppHeader
          testID="header"
          showViewToggle={true}
          onViewModeChange={onViewModeChange}
          viewMode="cards"
        />
      );
      fireEvent.press(getByTestId('header-list-btn'));
      expect(onViewModeChange).toHaveBeenCalledWith('list');
    });
  });
});
