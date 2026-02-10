/**
 * Secure storage utility for sensitive data using Web Crypto.
 * Falls back to in-memory storage when crypto is unavailable.
 */

const STORAGE_PREFIX = 'secure:';
const MEMORY_STORE = new Map<string, string>();

type CryptoKeyLike = CryptoKey;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

function hasWebCrypto(): boolean {
  return typeof crypto !== 'undefined' && typeof crypto.subtle !== 'undefined';
}

function encodeBase64(data: Uint8Array): string {
  if (typeof btoa === 'function') {
    let binary = '';
    data.forEach((byte) => {
      binary += String.fromCharCode(byte);
    });
    return btoa(binary);
  }
  return Buffer.from(data).toString('base64');
}

function decodeBase64(data: string): Uint8Array {
  if (typeof atob === 'function') {
    const binary = atob(data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
  return new Uint8Array(Buffer.from(data, 'base64'));
}

async function generateKey(): Promise<CryptoKeyLike> {
  return crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);
}

let cachedKey: CryptoKeyLike | null = null;

async function getKey(): Promise<CryptoKeyLike> {
  if (!cachedKey) {
    cachedKey = await generateKey();
  }
  return cachedKey;
}

function getStorage(): Storage | null {
  if (typeof sessionStorage === 'undefined') {
    return null;
  }
  return sessionStorage;
}

function getMemoryKey(key: string): string {
  return `${STORAGE_PREFIX}${key}`;
}

async function encryptValue(value: string): Promise<string> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await getKey();
  const encoded = textEncoder.encode(value);
  const encrypted = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, encoded));
  const payload = new Uint8Array(iv.length + encrypted.length);
  payload.set(iv, 0);
  payload.set(encrypted, iv.length);
  return encodeBase64(payload);
}

async function decryptValue(value: string): Promise<string | null> {
  try {
    const payload = decodeBase64(value);
    const iv = payload.slice(0, 12);
    const ciphertext = payload.slice(12);
    const key = await getKey();
    const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ciphertext);
    return textDecoder.decode(decrypted);
  } catch {
    return null;
  }
}

export const secureStorage = {
  async setItem(key: string, value: string): Promise<void> {
    const storageKey = getMemoryKey(key);
    if (!hasWebCrypto()) {
      MEMORY_STORE.set(storageKey, value);
      return;
    }

    const storage = getStorage();
    if (!storage) {
      MEMORY_STORE.set(storageKey, value);
      return;
    }

    const encrypted = await encryptValue(value);
    storage.setItem(storageKey, encrypted);
  },

  async getItem(key: string): Promise<string | null> {
    const storageKey = getMemoryKey(key);
    if (!hasWebCrypto()) {
      return MEMORY_STORE.get(storageKey) ?? null;
    }

    const storage = getStorage();
    if (!storage) {
      return MEMORY_STORE.get(storageKey) ?? null;
    }

    const encrypted = storage.getItem(storageKey);
    if (!encrypted) return null;
    return await decryptValue(encrypted);
  },

  removeItem(key: string): void {
    const storageKey = getMemoryKey(key);
    MEMORY_STORE.delete(storageKey);
    const storage = getStorage();
    storage?.removeItem(storageKey);
  },

  clear(): void {
    MEMORY_STORE.clear();
    const storage = getStorage();
    if (!storage) return;
    Object.keys(storage).forEach((key) => {
      if (key.startsWith(STORAGE_PREFIX)) {
        storage.removeItem(key);
      }
    });
  },
};
