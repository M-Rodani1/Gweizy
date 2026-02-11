export function enforceMaxLength(value: string, maxLength: number): string {
  if (maxLength <= 0 || !Number.isFinite(maxLength)) {
    return '';
  }
  if (value.length <= maxLength) {
    return value;
  }
  return value.slice(0, maxLength);
}

export function isWithinLength(value: string, maxLength: number): boolean {
  if (maxLength <= 0 || !Number.isFinite(maxLength)) {
    return false;
  }
  return value.length <= maxLength;
}
