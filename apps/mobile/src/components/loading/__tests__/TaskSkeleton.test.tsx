/**
 * TaskSkeleton Tests
 */

import React from 'react';
import { render } from '@testing-library/react-native';
import { TaskSkeleton } from '../TaskSkeleton';

describe('TaskSkeleton', () => {
  describe('rendering', () => {
    it('renders with default props', () => {
      const { getByTestId } = render(<TaskSkeleton />);
      expect(getByTestId('task-skeleton')).toBeTruthy();
    });

    it('renders with custom testID', () => {
      const { getByTestId } = render(<TaskSkeleton testID="custom-tasks" />);
      expect(getByTestId('custom-tasks')).toBeTruthy();
    });

    it('renders default count of 4 items', () => {
      const { getByTestId, queryByTestId } = render(<TaskSkeleton />);
      // Should have items 0-3
      expect(getByTestId('task-skeleton-item-0')).toBeTruthy();
      expect(getByTestId('task-skeleton-item-1')).toBeTruthy();
      expect(getByTestId('task-skeleton-item-2')).toBeTruthy();
      expect(getByTestId('task-skeleton-item-3')).toBeTruthy();
      // Should not have item 4
      expect(queryByTestId('task-skeleton-item-4')).toBeNull();
    });

    it('renders custom count of items', () => {
      const { getByTestId, queryByTestId } = render(<TaskSkeleton count={2} />);
      expect(getByTestId('task-skeleton-item-0')).toBeTruthy();
      expect(getByTestId('task-skeleton-item-1')).toBeTruthy();
      expect(queryByTestId('task-skeleton-item-2')).toBeNull();
    });
  });

  describe('task item structure', () => {
    it('renders checkbox skeleton', () => {
      const { getByTestId } = render(<TaskSkeleton count={1} />);
      expect(getByTestId('task-skeleton-item-0-checkbox')).toBeTruthy();
    });

    it('renders title skeleton', () => {
      const { getByTestId } = render(<TaskSkeleton count={1} />);
      expect(getByTestId('task-skeleton-item-0-title')).toBeTruthy();
    });

    it('renders due date skeleton', () => {
      const { getByTestId } = render(<TaskSkeleton count={1} />);
      expect(getByTestId('task-skeleton-item-0-due')).toBeTruthy();
    });

    it('renders priority badge skeleton', () => {
      const { getByTestId } = render(<TaskSkeleton count={1} />);
      expect(getByTestId('task-skeleton-item-0-priority')).toBeTruthy();
    });
  });

  describe('edge cases', () => {
    it('handles count of 0', () => {
      const { getByTestId, queryByTestId } = render(<TaskSkeleton count={0} />);
      expect(getByTestId('task-skeleton')).toBeTruthy();
      expect(queryByTestId('task-skeleton-item-0')).toBeNull();
    });

    it('handles count of 1', () => {
      const { getByTestId, queryByTestId } = render(<TaskSkeleton count={1} />);
      expect(getByTestId('task-skeleton-item-0')).toBeTruthy();
      expect(queryByTestId('task-skeleton-item-1')).toBeNull();
    });

    it('handles large count', () => {
      const { getByTestId } = render(<TaskSkeleton count={8} />);
      expect(getByTestId('task-skeleton-item-7')).toBeTruthy();
    });
  });
});
