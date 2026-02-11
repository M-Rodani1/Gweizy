import { readdirSync, statSync, readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const SRC_ROOT = path.resolve(process.cwd(), 'src');
const OUTPUT_DIR = path.resolve(process.cwd(), 'docs');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'api.md');

const walk = (dir, files = []) => {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stats = statSync(full);
    if (stats.isDirectory()) {
      if (entry === '__tests__') continue;
      walk(full, files);
      continue;
    }
    if (entry.endsWith('.ts') || entry.endsWith('.tsx')) {
      files.push(full);
    }
  }
  return files;
};

const extractExports = (contents) => {
  const exports = [];
  const named = contents.matchAll(/export\s+(?:const|function|class|type|interface|enum)\s+([A-Za-z0-9_]+)/g);
  for (const match of named) {
    exports.push(match[1]);
  }
  return exports;
};

const main = () => {
  const files = walk(SRC_ROOT);
  const entries = files.map((file) => {
    const contents = readFileSync(file, 'utf8');
    const exports = extractExports(contents);
    if (exports.length === 0) return null;
    return {
      file: path.relative(SRC_ROOT, file),
      exports
    };
  }).filter(Boolean);

  const lines = ['# API Reference', '', 'Generated from source exports.'];
  entries.forEach((entry) => {
    lines.push('', `## ${entry.file}`, '');
    entry.exports.forEach((name) => {
      lines.push(`- ${name}`);
    });
  });

  mkdirSync(OUTPUT_DIR, { recursive: true });
  writeFileSync(OUTPUT_FILE, `${lines.join('\n')}\n`, 'utf8');
  console.log(`API docs generated: ${OUTPUT_FILE}`);
};

main();
