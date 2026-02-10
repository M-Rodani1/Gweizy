import fs from 'node:fs';
import path from 'node:path';

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(2)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(2)} MB`;
}

export function collectBundleFiles(rootDir) {
  const entries = [];

  if (!fs.existsSync(rootDir)) {
    return entries;
  }

  const queue = [rootDir];
  while (queue.length > 0) {
    const current = queue.pop();
    if (!current) continue;

    const stat = fs.statSync(current);
    if (stat.isDirectory()) {
      const children = fs.readdirSync(current);
      for (const child of children) {
        queue.push(path.join(current, child));
      }
      continue;
    }

    entries.push({
      filePath: current,
      bytes: stat.size,
    });
  }

  return entries;
}

export function createBundleSizeReport(rootDir, options = {}) {
  const { topN = 10 } = options;
  const files = collectBundleFiles(rootDir)
    .sort((a, b) => b.bytes - a.bytes)
    .map((entry) => ({
      ...entry,
      formattedSize: formatBytes(entry.bytes),
    }));

  const totalBytes = files.reduce((sum, file) => sum + file.bytes, 0);
  const topFiles = files.slice(0, topN);

  return {
    rootDir,
    totalBytes,
    totalFormatted: formatBytes(totalBytes),
    fileCount: files.length,
    files,
    topFiles,
  };
}

export function printBundleReport(report, { json = false } = {}) {
  if (json) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }

  console.log(`Bundle size report for ${report.rootDir}`);
  console.log(`Total: ${report.totalFormatted} across ${report.fileCount} files`);
  if (report.topFiles.length > 0) {
    console.log('Largest files:');
    for (const entry of report.topFiles) {
      console.log(`- ${entry.formattedSize}  ${entry.filePath}`);
    }
  }
}

function parseArgs(argv) {
  const args = [...argv];
  let dir = process.env.BUNDLE_DIR || path.resolve(process.cwd(), 'dist/assets');
  let topN = 10;
  let json = false;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--dir' && args[i + 1]) {
      dir = path.resolve(args[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--top' && args[i + 1]) {
      topN = Number(args[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--json') {
      json = true;
    }
  }

  return { dir, topN, json };
}

function main() {
  const { dir, topN, json } = parseArgs(process.argv.slice(2));

  if (!fs.existsSync(dir)) {
    console.error(`Bundle directory not found: ${dir}`);
    process.exit(1);
  }

  const report = createBundleSizeReport(dir, { topN });
  printBundleReport(report, { json });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
