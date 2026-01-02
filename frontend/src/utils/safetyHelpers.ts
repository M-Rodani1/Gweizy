/**
 * Safety helper functions to prevent common runtime errors
 * Use these throughout your codebase to avoid "Cannot read property of undefined" errors
 */

/**
 * Safely access array length
 * @returns 0 if array is null/undefined, otherwise returns the length
 */
export const safeLength = (arr: any): number => {
  return Array.isArray(arr) ? arr.length : 0;
};

/**
 * Safely map over an array
 * @returns empty array if input is not an array
 */
export const safeMap = <T, U>(arr: any, fn: (item: T, index: number) => U): U[] => {
  return Array.isArray(arr) ? arr.map(fn) : [];
};

/**
 * Safely filter an array
 * @returns empty array if input is not an array
 */
export const safeFilter = <T>(arr: any, fn: (item: T, index: number) => boolean): T[] => {
  return Array.isArray(arr) ? arr.filter(fn) : [];
};

/**
 * Safely call toFixed on a number
 * @returns formatted string or fallback value
 */
export const safeToFixed = (
  value: any,
  decimals: number = 2,
  fallback: string = 'N/A'
): string => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return fallback;
  }
  return Number(value).toFixed(decimals);
};

/**
 * Safely access nested object properties
 * @example safeGet(data, 'user.profile.name', 'Unknown')
 */
export const safeGet = (obj: any, path: string, defaultValue: any = null): any => {
  if (!obj) return defaultValue;

  const keys = path.split('.');
  let result = obj;

  for (const key of keys) {
    if (result === null || result === undefined) {
      return defaultValue;
    }
    result = result[key];
  }

  return result !== undefined ? result : defaultValue;
};

/**
 * Safely format a number as USD
 */
export const safeFormatUSD = (value: any, decimals: number = 2): string => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return '$0.00';
  }
  return `$${Number(value).toFixed(decimals)}`;
};

/**
 * Safely format a number as percentage
 */
export const safeFormatPercent = (value: any, decimals: number = 1): string => {
  if (value === null || value === undefined || isNaN(Number(value))) {
    return '0%';
  }
  return `${Number(value).toFixed(decimals)}%`;
};

/**
 * Ensure a value is an array
 */
export const ensureArray = <T>(value: any): T[] => {
  return Array.isArray(value) ? value : [];
};

/**
 * Safely get first item from array
 */
export const safeFirst = <T>(arr: any, fallback?: T): T | undefined => {
  return Array.isArray(arr) && arr.length > 0 ? arr[0] : fallback;
};

/**
 * Safely get last item from array
 */
export const safeLast = <T>(arr: any, fallback?: T): T | undefined => {
  return Array.isArray(arr) && arr.length > 0 ? arr[arr.length - 1] : fallback;
};

/**
 * Check if value is null or undefined
 */
export const isNullish = (value: any): boolean => {
  return value === null || value === undefined;
};

/**
 * Provide default value if nullish
 */
export const defaultIfNullish = <T>(value: any, defaultValue: T): T => {
  return isNullish(value) ? defaultValue : value;
};
