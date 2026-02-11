export function getSecureRandomBytes(length: number): Uint8Array {
  if (length <= 0 || !Number.isFinite(length)) {
    throw new Error('Length must be a positive number');
  }

  if (typeof crypto === 'undefined' || typeof crypto.getRandomValues !== 'function') {
    throw new Error('Secure random generator unavailable');
  }

  return crypto.getRandomValues(new Uint8Array(length));
}

export function getSecureRandomHex(length: number): string {
  const bytes = getSecureRandomBytes(length);
  return Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}
