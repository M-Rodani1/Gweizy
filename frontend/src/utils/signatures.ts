const SIGNATURE_REGEX = /^0x[0-9a-fA-F]{130}$/;

export interface ParsedSignature {
  r: string;
  s: string;
  v: number;
}

export function normalizeSignatureV(v: number): number {
  if (v === 0 || v === 1) return v + 27;
  return v;
}

export function parseSignature(signature: string): ParsedSignature | null {
  if (!SIGNATURE_REGEX.test(signature)) {
    return null;
  }

  const hex = signature.slice(2);
  const r = `0x${hex.slice(0, 64)}`;
  const s = `0x${hex.slice(64, 128)}`;
  const v = parseInt(hex.slice(128, 130), 16);

  if (Number.isNaN(v)) {
    return null;
  }

  return { r, s, v: normalizeSignatureV(v) };
}

export function isValidSignatureFormat(signature: string): boolean {
  const parsed = parseSignature(signature);
  if (!parsed) return false;
  return parsed.v === 27 || parsed.v === 28;
}
