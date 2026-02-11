const RATE_IN_BYTES = 136;
const ROUNDS = 24;

const ROTATION_OFFSETS = [
  0, 1, 62, 28, 27,
  36, 44, 6, 55, 20,
  3, 10, 43, 25, 39,
  41, 45, 15, 21, 8,
  18, 2, 61, 56, 14,
];

const ROUND_CONSTANTS = [
  0x0000000000000001n,
  0x0000000000008082n,
  0x800000000000808an,
  0x8000000080008000n,
  0x000000000000808bn,
  0x0000000080000001n,
  0x8000000080008081n,
  0x8000000000008009n,
  0x000000000000008an,
  0x0000000000000088n,
  0x0000000080008009n,
  0x000000008000000an,
  0x000000008000808bn,
  0x800000000000008bn,
  0x8000000000008089n,
  0x8000000000008003n,
  0x8000000000008002n,
  0x8000000000000080n,
  0x000000000000800an,
  0x800000008000000an,
  0x8000000080008081n,
  0x8000000000008080n,
  0x0000000080000001n,
  0x8000000080008008n,
];

const MASK_64 = (1n << 64n) - 1n;

function rotl64(value: bigint, shift: number): bigint {
  const s = BigInt(shift % 64);
  return ((value << s) | (value >> (64n - s))) & MASK_64;
}

function keccakF(state: bigint[]): void {
  for (let round = 0; round < ROUNDS; round += 1) {
    const c = new Array<bigint>(5);
    const d = new Array<bigint>(5);

    for (let x = 0; x < 5; x += 1) {
      c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }

    for (let x = 0; x < 5; x += 1) {
      d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
    }

    for (let x = 0; x < 5; x += 1) {
      for (let y = 0; y < 5; y += 1) {
        state[x + 5 * y] = (state[x + 5 * y] ^ d[x]) & MASK_64;
      }
    }

    const b = new Array<bigint>(25);
    for (let x = 0; x < 5; x += 1) {
      for (let y = 0; y < 5; y += 1) {
        const idx = x + 5 * y;
        const rot = ROTATION_OFFSETS[idx];
        const newX = y;
        const newY = (2 * x + 3 * y) % 5;
        b[newX + 5 * newY] = rotl64(state[idx], rot);
      }
    }

    for (let y = 0; y < 5; y += 1) {
      for (let x = 0; x < 5; x += 1) {
        const notNext = (~b[(x + 1) % 5 + 5 * y]) & MASK_64;
        state[x + 5 * y] = (b[x + 5 * y] ^ (notNext & b[(x + 2) % 5 + 5 * y])) & MASK_64;
      }
    }

    state[0] = (state[0] ^ ROUND_CONSTANTS[round]) & MASK_64;
  }
}

function stringToBytes(input: string): Uint8Array {
  const bytes = new Uint8Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    bytes[i] = input.charCodeAt(i);
  }
  return bytes;
}

export function keccak256(message: string): string {
  const state = new Array<bigint>(25).fill(0n);
  const bytes = stringToBytes(message);

  let offset = 0;
  while (offset < bytes.length) {
    const blockSize = Math.min(RATE_IN_BYTES, bytes.length - offset);

    for (let i = 0; i < blockSize; i += 1) {
      const laneIndex = Math.floor(i / 8);
      const shift = BigInt((i % 8) * 8);
      state[laneIndex] ^= BigInt(bytes[offset + i]) << shift;
    }

    offset += blockSize;

    if (blockSize === RATE_IN_BYTES) {
      keccakF(state);
    } else {
      break;
    }
  }

  const paddingIndex = offset % RATE_IN_BYTES;
  const laneIndex = Math.floor(paddingIndex / 8);
  const shift = BigInt((paddingIndex % 8) * 8);
  state[laneIndex] ^= 0x01n << shift;
  state[Math.floor((RATE_IN_BYTES - 1) / 8)] ^= 0x80n << BigInt(((RATE_IN_BYTES - 1) % 8) * 8);

  keccakF(state);

  let output = '';
  for (let i = 0; i < 32; i += 1) {
    const lane = state[Math.floor(i / 8)];
    const shiftOut = BigInt((i % 8) * 8);
    const byte = Number((lane >> shiftOut) & 0xffn);
    output += byte.toString(16).padStart(2, '0');
  }

  return output;
}
