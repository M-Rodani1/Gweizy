/**
 * VirtualizedList Component Tests
 *
 * Tests for the virtualized list component that efficiently renders
 * long lists by only rendering visible items plus an overscan buffer.
 */

import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import VirtualizedList from '../../components/ui/VirtualizedList';

// ============================================================================
// Test Helpers
// ============================================================================

interface TestItem {
  id: number;
  label: string;
}

const createTestItems = (count: number): TestItem[] =>
  Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    label: `Item ${i + 1}`,
  }));

const defaultRenderItem = (item: TestItem) => (
  <div data-testid={`item-${item.id}`}>{item.label}</div>
);

const defaultGetKey = (item: TestItem) => item.id;

// ============================================================================
// Tests
// ============================================================================

describe('VirtualizedList', () => {
  describe('Rendering', () => {
    it('should render nothing when items array is empty', () => {
      const { container } = render(
        <VirtualizedList
          items={[]}
          itemHeight={40}
          maxHeight={400}
          renderItem={defaultRenderItem}
        />
      );

      expect(container.firstChild).toBeNull();
    });

    it('should render visible items', () => {
      const items = createTestItems(100);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // With maxHeight=200 and itemHeight=40, about 5 items should be visible
      // Plus overscan of 3, so items 1-8 should be rendered
      expect(screen.getByTestId('item-1')).toBeInTheDocument();
      expect(screen.getByTestId('item-5')).toBeInTheDocument();
    });

    it('should not render items beyond visible range plus overscan', () => {
      const items = createTestItems(100);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          overscan={2}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // Items far down the list should not be rendered
      expect(screen.queryByTestId('item-50')).not.toBeInTheDocument();
      expect(screen.queryByTestId('item-100')).not.toBeInTheDocument();
    });

    it('should render all items when list fits within maxHeight', () => {
      const items = createTestItems(5);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={400}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // All 5 items should be rendered
      items.forEach((item) => {
        expect(screen.getByTestId(`item-${item.id}`)).toBeInTheDocument();
      });
    });
  });

  describe('Scroll Behavior', () => {
    it('should update visible items on scroll', () => {
      const items = createTestItems(100);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          overscan={0}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      const scrollContainer = container.firstChild as HTMLElement;

      // Initially, items 1-5 should be visible (maxHeight=200, itemHeight=40)
      expect(screen.getByTestId('item-1')).toBeInTheDocument();

      // Scroll down by 400px (10 items)
      fireEvent.scroll(scrollContainer, { target: { scrollTop: 400 } });

      // Now items 11-15 should be visible
      expect(screen.getByTestId('item-11')).toBeInTheDocument();
      expect(screen.getByTestId('item-15')).toBeInTheDocument();
    });

    it('should handle scroll to top', () => {
      const items = createTestItems(100);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          overscan={0}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      const scrollContainer = container.firstChild as HTMLElement;

      // Scroll down first
      fireEvent.scroll(scrollContainer, { target: { scrollTop: 1000 } });

      // Scroll back to top
      fireEvent.scroll(scrollContainer, { target: { scrollTop: 0 } });

      // First items should be visible again
      expect(screen.getByTestId('item-1')).toBeInTheDocument();
    });

    it('should handle scroll to bottom', () => {
      const items = createTestItems(100);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          overscan={0}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      const scrollContainer = container.firstChild as HTMLElement;

      // Scroll to bottom (total height = 100 * 40 = 4000, visible = 200)
      fireEvent.scroll(scrollContainer, { target: { scrollTop: 3800 } });

      // Last items should be visible
      expect(screen.getByTestId('item-100')).toBeInTheDocument();
    });
  });

  describe('Overscan', () => {
    it('should render extra items based on overscan value', () => {
      const items = createTestItems(50);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          overscan={5}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // With maxHeight=200 and itemHeight=40, ~5 items are visible
      // With overscan=5, items 1-10 should be rendered (5 visible + 5 after)
      expect(screen.getByTestId('item-1')).toBeInTheDocument();
      expect(screen.getByTestId('item-10')).toBeInTheDocument();
    });

    it('should use default overscan of 3', () => {
      const items = createTestItems(50);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // 5 visible + 3 overscan after = 8 items
      expect(screen.getByTestId('item-8')).toBeInTheDocument();
    });

    it('should handle zero overscan', () => {
      const items = createTestItems(50);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          overscan={0}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // With maxHeight=200, itemHeight=40, endIndex = floor(200/40) = 5
      // So items at indices 0-5 (6 items: Item 1 through Item 6) are rendered
      expect(screen.getByTestId('item-6')).toBeInTheDocument();
      expect(screen.queryByTestId('item-7')).not.toBeInTheDocument();
    });
  });

  describe('Item Positioning', () => {
    it('should position items correctly with absolute positioning', () => {
      const items = createTestItems(10);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={50}
          maxHeight={500}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // Get all item wrappers
      const itemWrappers = container.querySelectorAll('[style*="position: absolute"]');

      // Check first item position
      expect(itemWrappers[0]).toHaveStyle('top: 0px');
      expect(itemWrappers[0]).toHaveStyle('height: 50px');

      // Check second item position
      expect(itemWrappers[1]).toHaveStyle('top: 50px');

      // Check third item position
      expect(itemWrappers[2]).toHaveStyle('top: 100px');
    });

    it('should set correct container height', () => {
      const items = createTestItems(100);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={300}
          renderItem={defaultRenderItem}
        />
      );

      const scrollContainer = container.firstChild as HTMLElement;
      expect(scrollContainer).toHaveStyle('height: 300px');
    });

    it('should set total content height for scrolling', () => {
      const items = createTestItems(100);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={300}
          renderItem={defaultRenderItem}
        />
      );

      const contentContainer = container.querySelector('[style*="position: relative"]') as HTMLElement;
      // Total height = 100 items * 40px = 4000px
      expect(contentContainer).toHaveStyle('height: 4000px');
    });
  });

  describe('Custom Key Function', () => {
    it('should use custom getKey function for item keys', () => {
      const items = createTestItems(5);
      const getKey = vi.fn((item: TestItem) => `custom-${item.id}`);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={400}
          renderItem={defaultRenderItem}
          getKey={getKey}
        />
      );

      expect(getKey).toHaveBeenCalledTimes(5);
      expect(getKey).toHaveBeenCalledWith(items[0], 0);
      expect(getKey).toHaveBeenCalledWith(items[4], 4);
    });

    it('should use index as key when getKey is not provided', () => {
      const items = createTestItems(3);

      // This should not throw
      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={400}
          renderItem={defaultRenderItem}
        />
      );

      // Items should still render
      expect(screen.getByTestId('item-1')).toBeInTheDocument();
    });
  });

  describe('CSS Classes', () => {
    it('should apply custom className to scroll container', () => {
      const items = createTestItems(5);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={400}
          className="custom-scroll-class"
          renderItem={defaultRenderItem}
        />
      );

      expect(container.firstChild).toHaveClass('custom-scroll-class');
    });

    it('should apply contentClassName to inner content container', () => {
      const items = createTestItems(5);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={400}
          contentClassName="custom-content-class"
          renderItem={defaultRenderItem}
        />
      );

      const contentContainer = container.querySelector('.custom-content-class');
      expect(contentContainer).toBeInTheDocument();
    });
  });

  describe('Dynamic Items', () => {
    it('should reset scroll position when items change', () => {
      const items1 = createTestItems(50);
      const items2 = createTestItems(100);

      const { rerender, container } = render(
        <VirtualizedList
          items={items1}
          itemHeight={40}
          maxHeight={200}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      const scrollContainer = container.firstChild as HTMLElement;

      // Scroll down
      fireEvent.scroll(scrollContainer, { target: { scrollTop: 500 } });

      // Change items
      rerender(
        <VirtualizedList
          items={items2}
          itemHeight={40}
          maxHeight={200}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // First items should be visible after reset
      expect(screen.getByTestId('item-1')).toBeInTheDocument();
    });

    it('should handle items being added', () => {
      const items1 = createTestItems(5);
      const items2 = createTestItems(10);

      const { rerender } = render(
        <VirtualizedList
          items={items1}
          itemHeight={40}
          maxHeight={400}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      expect(screen.getByTestId('item-5')).toBeInTheDocument();
      expect(screen.queryByTestId('item-6')).not.toBeInTheDocument();

      rerender(
        <VirtualizedList
          items={items2}
          itemHeight={40}
          maxHeight={400}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      expect(screen.getByTestId('item-6')).toBeInTheDocument();
      expect(screen.getByTestId('item-10')).toBeInTheDocument();
    });

    it('should handle items being removed', () => {
      const items1 = createTestItems(10);
      const items2 = createTestItems(3);

      const { rerender } = render(
        <VirtualizedList
          items={items1}
          itemHeight={40}
          maxHeight={500}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      expect(screen.getByTestId('item-10')).toBeInTheDocument();

      rerender(
        <VirtualizedList
          items={items2}
          itemHeight={40}
          maxHeight={500}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      expect(screen.queryByTestId('item-10')).not.toBeInTheDocument();
      expect(screen.getByTestId('item-3')).toBeInTheDocument();
    });
  });

  describe('Render Function', () => {
    it('should pass item and index to renderItem', () => {
      const items = createTestItems(5);
      const renderItem = vi.fn((item: TestItem, index: number) => (
        <div data-testid={`item-${item.id}`}>
          {item.label} at {index}
        </div>
      ));

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={400}
          renderItem={renderItem}
        />
      );

      expect(renderItem).toHaveBeenCalledTimes(5);
      expect(renderItem).toHaveBeenCalledWith(items[0], 0);
      expect(renderItem).toHaveBeenCalledWith(items[2], 2);
      expect(renderItem).toHaveBeenCalledWith(items[4], 4);
    });

    it('should handle complex render items', () => {
      interface ComplexItem {
        id: number;
        title: string;
        description: string;
        status: 'active' | 'inactive';
      }

      const items: ComplexItem[] = [
        { id: 1, title: 'First', description: 'First item', status: 'active' },
        { id: 2, title: 'Second', description: 'Second item', status: 'inactive' },
      ];

      const renderItem = (item: ComplexItem) => (
        <div data-testid={`item-${item.id}`}>
          <h3>{item.title}</h3>
          <p>{item.description}</p>
          <span>{item.status}</span>
        </div>
      );

      render(
        <VirtualizedList
          items={items}
          itemHeight={100}
          maxHeight={400}
          renderItem={renderItem}
          getKey={(item) => item.id}
        />
      );

      expect(screen.getByText('First')).toBeInTheDocument();
      expect(screen.getByText('First item')).toBeInTheDocument();
      expect(screen.getByText('active')).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle single item', () => {
      const items = createTestItems(1);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={400}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      expect(screen.getByTestId('item-1')).toBeInTheDocument();
    });

    it('should handle very large item heights', () => {
      const items = createTestItems(10);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={500}
          maxHeight={300}
          overscan={0}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // Only 1 item fits in viewport, so only item 1 should be rendered
      expect(screen.getByTestId('item-1')).toBeInTheDocument();

      const contentContainer = container.querySelector('[style*="position: relative"]') as HTMLElement;
      expect(contentContainer).toHaveStyle('height: 5000px');
    });

    it('should handle very small item heights', () => {
      const items = createTestItems(1000);

      render(
        <VirtualizedList
          items={items}
          itemHeight={10}
          maxHeight={100}
          overscan={2}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      // With maxHeight=100, itemHeight=10: endIndex = floor(100/10) + 2 = 12
      // So items at indices 0-12 (13 items: Item 1 through Item 13) are rendered
      expect(screen.getByTestId('item-1')).toBeInTheDocument();
      expect(screen.getByTestId('item-13')).toBeInTheDocument();
      expect(screen.queryByTestId('item-14')).not.toBeInTheDocument();
    });

    it('should handle maxHeight smaller than item height', () => {
      const items = createTestItems(5);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={100}
          maxHeight={50}
          overscan={0}
          renderItem={defaultRenderItem}
          getKey={defaultGetKey}
        />
      );

      const scrollContainer = container.firstChild as HTMLElement;
      expect(scrollContainer).toHaveStyle('height: 50px');
      expect(scrollContainer).toHaveStyle('overflow-y: auto');
    });

    it('should handle items with undefined values gracefully', () => {
      interface NullableItem {
        id: number;
        value: string | null;
      }

      const items: NullableItem[] = [
        { id: 1, value: 'test' },
        { id: 2, value: null },
        { id: 3, value: 'another' },
      ];

      const renderItem = (item: NullableItem) => (
        <div data-testid={`item-${item.id}`}>{item.value ?? 'N/A'}</div>
      );

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={400}
          renderItem={renderItem}
          getKey={(item) => item.id}
        />
      );

      expect(screen.getByText('test')).toBeInTheDocument();
      expect(screen.getByText('N/A')).toBeInTheDocument();
      expect(screen.getByText('another')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('should only render visible items for large lists', () => {
      const items = createTestItems(10000);
      const renderItem = vi.fn(defaultRenderItem);

      render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          overscan={3}
          renderItem={renderItem}
          getKey={defaultGetKey}
        />
      );

      // With maxHeight=200, itemHeight=40, overscan=3:
      // endIndex = floor(200/40) + 3 = 5 + 3 = 8
      // So items at indices 0-8 (9 items) are rendered
      expect(renderItem).toHaveBeenCalledTimes(9);
    });

    it('should maintain performance on scroll', () => {
      const items = createTestItems(10000);
      const renderItem = vi.fn(defaultRenderItem);

      const { container } = render(
        <VirtualizedList
          items={items}
          itemHeight={40}
          maxHeight={200}
          overscan={3}
          renderItem={renderItem}
          getKey={defaultGetKey}
        />
      );

      const scrollContainer = container.firstChild as HTMLElement;

      // Scroll multiple times
      for (let i = 0; i < 10; i++) {
        fireEvent.scroll(scrollContainer, { target: { scrollTop: i * 1000 } });
      }

      // Should only render visible items each time, not the entire list
      // Each scroll renders about 8-11 items depending on position
      expect(renderItem.mock.calls.length).toBeLessThan(200);
    });
  });
});
