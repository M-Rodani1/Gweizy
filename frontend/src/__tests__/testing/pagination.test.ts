import { describe, it, expect } from 'vitest';
import { paginate } from '../../utils/pagination';

describe('Pagination logic', () => {
  it('returns a page of items with metadata', () => {
    const items = Array.from({ length: 10 }, (_, index) => index + 1);
    const result = paginate(items, 2, 3);

    expect(result.items).toEqual([4, 5, 6]);
    expect(result.page).toBe(2);
    expect(result.pageSize).toBe(3);
    expect(result.totalItems).toBe(10);
    expect(result.totalPages).toBe(4);
    expect(result.hasPrev).toBe(true);
    expect(result.hasNext).toBe(true);
  });

  it('clamps page numbers within valid bounds', () => {
    const items = Array.from({ length: 5 }, (_, index) => index + 1);

    const firstPage = paginate(items, -2, 2);
    expect(firstPage.page).toBe(1);
    expect(firstPage.items).toEqual([1, 2]);

    const lastPage = paginate(items, 99, 2);
    expect(lastPage.page).toBe(3);
    expect(lastPage.items).toEqual([5]);
    expect(lastPage.hasNext).toBe(false);
  });

  it('defaults invalid page sizes to 1', () => {
    const items = ['a', 'b', 'c'];
    const result = paginate(items, 1, 0);

    expect(result.pageSize).toBe(1);
    expect(result.items).toEqual(['a']);
    expect(result.totalPages).toBe(3);
  });

  it('handles empty collections gracefully', () => {
    const result = paginate([], 1, 5);

    expect(result.items).toEqual([]);
    expect(result.totalItems).toBe(0);
    expect(result.totalPages).toBe(0);
    expect(result.hasNext).toBe(false);
    expect(result.hasPrev).toBe(false);
  });
});
