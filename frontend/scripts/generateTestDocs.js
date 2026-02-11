import { readdirSync, statSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const TEST_ROOT = path.resolve(process.cwd(), 'src/__tests__');
const OUTPUT_DIR = path.resolve(process.cwd(), 'docs');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'tests.md');

const walk = (dir, files = []) => {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stats = statSync(full);
    if (stats.isDirectory()) {
      walk(full, files);
      continue;
    }
    if (entry.endsWith('.test.ts') || entry.endsWith('.test.tsx')) {
      files.push(full);
    }
  }
  return files;
};

const main = () => {
  const files = walk(TEST_ROOT);
  const lines = ['# Test Reference', '', 'Generated from test files.'];

  files.forEach((file) => {
    lines.push(`- ${path.relative(TEST_ROOT, file)}`);
  });

  mkdirSync(OUTPUT_DIR, { recursive: true });
  writeFileSync(OUTPUT_FILE, `${lines.join('\n')}\n`, 'utf8');
  console.log(`Test docs generated: ${OUTPUT_FILE}`);
};

main();
