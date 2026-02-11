import { keccak256 } from './keccak256';

const ADDRESS_REGEX = /^0x[0-9a-fA-F]{40}$/;

export function isValidAddress(address: string): boolean {
  return ADDRESS_REGEX.test(address);
}

export function toChecksumAddress(address: string): string {
  if (!isValidAddress(address)) {
    throw new Error('Invalid address format');
  }

  const normalized = address.toLowerCase().replace(/^0x/, '');
  const hash = keccak256(normalized);
  let checksummed = '0x';

  for (let i = 0; i < normalized.length; i += 1) {
    const char = normalized[i];
    const hashNibble = parseInt(hash[i], 16);

    if (Number.isNaN(hashNibble)) {
      checksummed += char;
      continue;
    }

    if (hashNibble >= 8) {
      checksummed += char.toUpperCase();
    } else {
      checksummed += char;
    }
  }

  return checksummed;
}

export function isChecksumAddress(address: string): boolean {
  if (!isValidAddress(address)) {
    return false;
  }

  try {
    return toChecksumAddress(address) === address;
  } catch {
    return false;
  }
}
