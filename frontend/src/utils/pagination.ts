export interface PaginationResult<T> {
  items: T[];
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

export function paginate<T>(items: T[], page: number, pageSize: number): PaginationResult<T> {
  const safePageSize = Math.max(1, Math.floor(pageSize) || 1);
  const totalItems = items.length;
  const totalPages = totalItems === 0 ? 0 : Math.ceil(totalItems / safePageSize);
  const safePage = totalPages === 0 ? 1 : Math.min(Math.max(1, Math.floor(page) || 1), totalPages);
  const start = totalPages === 0 ? 0 : (safePage - 1) * safePageSize;
  const pageItems = items.slice(start, start + safePageSize);

  return {
    items: pageItems,
    page: safePage,
    pageSize: safePageSize,
    totalItems,
    totalPages,
    hasNext: totalPages > 0 && safePage < totalPages,
    hasPrev: totalPages > 0 && safePage > 1,
  };
}
