import { readdirSync, statSync, readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const COMPONENT_ROOT = path.resolve(process.cwd(), 'src/components');
const OUTPUT_DIR = path.resolve(process.cwd(), 'docs');
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'components.md');

const walk = (dir, files = []) => {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stats = statSync(full);
    if (stats.isDirectory()) {
      if (entry === '__tests__') continue;
      walk(full, files);
      continue;
    }
    if (entry.endsWith('.tsx')) files.push(full);
  }
  return files;
};

const extractComponentName = (contents, fallback) => {
  const match = contents.match(/const\s+([A-Za-z0-9_]+)\s*:\s*React\.FC/);
  if (match) return match[1];
  const defaultMatch = contents.match(/export\s+default\s+([A-Za-z0-9_]+)/);
  if (defaultMatch) return defaultMatch[1];
  return fallback;
};

const main = () => {
  const files = walk(COMPONENT_ROOT);
  const lines = ['# Component Reference', '', 'Generated from component sources.'];

  files.forEach((file) => {
    const contents = readFileSync(file, 'utf8');
    const fallback = path.basename(file, path.extname(file));
    const name = extractComponentName(contents, fallback);
    lines.push('', `## ${name}`, '', `Source: \\`src/components/${path.relative(COMPONENT_ROOT, file)}\\``);
  });

  mkdirSync(OUTPUT_DIR, { recursive: true });
  writeFileSync(OUTPUT_FILE, `${lines.join('\n')}\n`, 'utf8');
  console.log(`Component docs generated: ${OUTPUT_FILE}`);
};

main();
