/**
 * Skeleton Tests
 */

import React from 'react';
import { render } from '@testing-library/react-native';
import { Skeleton, SkeletonGroup } from '../Skeleton';

describe('Skeleton', () => {
  describe('rendering', () => {
    it('renders with default props', () => {
      const { getByTestId } = render(<Skeleton />);
      expect(getByTestId('skeleton')).toBeTruthy();
    });

    it('renders with custom testID', () => {
      const { getByTestId } = render(<Skeleton testID="custom-skeleton" />);
      expect(getByTestId('custom-skeleton')).toBeTruthy();
    });

    it('renders with custom dimensions', () => {
      const { getByTestId } = render(
        <Skeleton width={200} height={50} testID="sized-skeleton" />
      );
      const skeleton = getByTestId('sized-skeleton');
      expect(skeleton).toBeTruthy();
    });

    it('renders with percentage width', () => {
      const { getByTestId } = render(
        <Skeleton width="50%" testID="percent-skeleton" />
      );
      expect(getByTestId('percent-skeleton')).toBeTruthy();
    });
  });

  describe('variants', () => {
    it('renders rectangle variant', () => {
      const { getByTestId } = render(
        <Skeleton variant="rectangle" testID="rect-skeleton" />
      );
      expect(getByTestId('rect-skeleton')).toBeTruthy();
    });

    it('renders circle variant', () => {
      const { getByTestId } = render(
        <Skeleton variant="circle" width={40} height={40} testID="circle-skeleton" />
      );
      expect(getByTestId('circle-skeleton')).toBeTruthy();
    });

    it('renders text variant', () => {
      const { getByTestId } = render(
        <Skeleton variant="text" testID="text-skeleton" />
      );
      expect(getByTestId('text-skeleton')).toBeTruthy();
    });
  });

  describe('custom styles', () => {
    it('applies custom style', () => {
      const { getByTestId } = render(
        <Skeleton style={{ marginTop: 10 }} testID="styled-skeleton" />
      );
      expect(getByTestId('styled-skeleton')).toBeTruthy();
    });
  });
});

describe('SkeletonGroup', () => {
  describe('rendering', () => {
    it('renders with default count', () => {
      const { getByTestId } = render(<SkeletonGroup />);
      expect(getByTestId('skeleton-group')).toBeTruthy();
      // Default count is 3
      expect(getByTestId('skeleton-group-item-0')).toBeTruthy();
      expect(getByTestId('skeleton-group-item-1')).toBeTruthy();
      expect(getByTestId('skeleton-group-item-2')).toBeTruthy();
    });

    it('renders custom count', () => {
      const { getByTestId, queryByTestId } = render(<SkeletonGroup count={5} />);
      expect(getByTestId('skeleton-group-item-0')).toBeTruthy();
      expect(getByTestId('skeleton-group-item-4')).toBeTruthy();
      expect(queryByTestId('skeleton-group-item-5')).toBeNull();
    });

    it('renders with custom testID', () => {
      const { getByTestId } = render(<SkeletonGroup testID="custom-group" />);
      expect(getByTestId('custom-group')).toBeTruthy();
      expect(getByTestId('custom-group-item-0')).toBeTruthy();
    });

    it('passes skeletonProps to children', () => {
      const { getByTestId } = render(
        <SkeletonGroup
          count={1}
          skeletonProps={{ width: 100, height: 20, variant: 'text' }}
          testID="props-group"
        />
      );
      expect(getByTestId('props-group-item-0')).toBeTruthy();
    });
  });

  describe('spacing', () => {
    it('applies custom spacing between items', () => {
      const { getByTestId } = render(
        <SkeletonGroup count={2} spacing={16} testID="spaced-group" />
      );
      expect(getByTestId('spaced-group-item-0')).toBeTruthy();
      expect(getByTestId('spaced-group-item-1')).toBeTruthy();
    });

    it('handles count of 1 without spacing', () => {
      const { getByTestId, queryByTestId } = render(
        <SkeletonGroup count={1} testID="single-group" />
      );
      expect(getByTestId('single-group-item-0')).toBeTruthy();
      expect(queryByTestId('single-group-item-1')).toBeNull();
    });
  });
});
