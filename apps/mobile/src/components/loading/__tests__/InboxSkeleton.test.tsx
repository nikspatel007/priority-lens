/**
 * InboxSkeleton Tests
 */

import React from 'react';
import { render } from '@testing-library/react-native';
import { InboxSkeleton } from '../InboxSkeleton';

describe('InboxSkeleton', () => {
  describe('rendering', () => {
    it('renders with default props', () => {
      const { getByTestId } = render(<InboxSkeleton />);
      expect(getByTestId('inbox-skeleton')).toBeTruthy();
    });

    it('renders with custom testID', () => {
      const { getByTestId } = render(<InboxSkeleton testID="custom-inbox" />);
      expect(getByTestId('custom-inbox')).toBeTruthy();
    });

    it('renders default count of 5 items', () => {
      const { getByTestId, queryByTestId } = render(<InboxSkeleton />);
      // Should have items 0-4
      expect(getByTestId('inbox-skeleton-item-0')).toBeTruthy();
      expect(getByTestId('inbox-skeleton-item-1')).toBeTruthy();
      expect(getByTestId('inbox-skeleton-item-2')).toBeTruthy();
      expect(getByTestId('inbox-skeleton-item-3')).toBeTruthy();
      expect(getByTestId('inbox-skeleton-item-4')).toBeTruthy();
      // Should not have item 5
      expect(queryByTestId('inbox-skeleton-item-5')).toBeNull();
    });

    it('renders custom count of items', () => {
      const { getByTestId, queryByTestId } = render(<InboxSkeleton count={3} />);
      expect(getByTestId('inbox-skeleton-item-0')).toBeTruthy();
      expect(getByTestId('inbox-skeleton-item-1')).toBeTruthy();
      expect(getByTestId('inbox-skeleton-item-2')).toBeTruthy();
      expect(queryByTestId('inbox-skeleton-item-3')).toBeNull();
    });
  });

  describe('email item structure', () => {
    it('renders avatar skeleton', () => {
      const { getByTestId } = render(<InboxSkeleton count={1} />);
      expect(getByTestId('inbox-skeleton-item-0-avatar')).toBeTruthy();
    });

    it('renders sender skeleton', () => {
      const { getByTestId } = render(<InboxSkeleton count={1} />);
      expect(getByTestId('inbox-skeleton-item-0-sender')).toBeTruthy();
    });

    it('renders time skeleton', () => {
      const { getByTestId } = render(<InboxSkeleton count={1} />);
      expect(getByTestId('inbox-skeleton-item-0-time')).toBeTruthy();
    });

    it('renders subject skeleton', () => {
      const { getByTestId } = render(<InboxSkeleton count={1} />);
      expect(getByTestId('inbox-skeleton-item-0-subject')).toBeTruthy();
    });

    it('renders preview skeleton', () => {
      const { getByTestId } = render(<InboxSkeleton count={1} />);
      expect(getByTestId('inbox-skeleton-item-0-preview')).toBeTruthy();
    });
  });

  describe('edge cases', () => {
    it('handles count of 0', () => {
      const { getByTestId, queryByTestId } = render(<InboxSkeleton count={0} />);
      expect(getByTestId('inbox-skeleton')).toBeTruthy();
      expect(queryByTestId('inbox-skeleton-item-0')).toBeNull();
    });

    it('handles count of 1', () => {
      const { getByTestId, queryByTestId } = render(<InboxSkeleton count={1} />);
      expect(getByTestId('inbox-skeleton-item-0')).toBeTruthy();
      expect(queryByTestId('inbox-skeleton-item-1')).toBeNull();
    });

    it('handles large count', () => {
      const { getByTestId } = render(<InboxSkeleton count={10} />);
      expect(getByTestId('inbox-skeleton-item-9')).toBeTruthy();
    });
  });
});
