import fs from 'node:fs';
import path from 'node:path';
import { createBundleSizeReport, formatBytes } from './bundleSizeReport.js';

export function evaluateBundleBudget(report, budget) {
  const failures = [];

  if (budget.totalMaxBytes && report.totalBytes > budget.totalMaxBytes) {
    failures.push(
      `Total bundle size ${formatBytes(report.totalBytes)} exceeds budget ${formatBytes(budget.totalMaxBytes)}`
    );
  }

  const largest = report.topFiles[0];
  if (largest && budget.largestAssetMaxBytes && largest.bytes > budget.largestAssetMaxBytes) {
    failures.push(
      `Largest asset ${largest.filePath} (${formatBytes(largest.bytes)}) exceeds budget ${formatBytes(budget.largestAssetMaxBytes)}`
    );
  }

  return { ok: failures.length === 0, failures };
}

function resolveBudgetFile(argPath) {
  if (argPath) {
    return path.resolve(argPath);
  }
  const envPath = process.env.BUNDLE_BUDGET_FILE;
  if (envPath) {
    return path.resolve(envPath);
  }
  return path.resolve(process.cwd(), 'bundle-budgets.json');
}

function parseArgs(argv) {
  const args = [...argv];
  let dir = process.env.BUNDLE_DIR || path.resolve(process.cwd(), 'dist/assets');
  let budgetFile;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === '--dir' && args[i + 1]) {
      dir = path.resolve(args[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--budget' && args[i + 1]) {
      budgetFile = args[i + 1];
      i += 1;
    }
  }

  return { dir, budgetFile: resolveBudgetFile(budgetFile) };
}

function main() {
  const { dir, budgetFile } = parseArgs(process.argv.slice(2));

  if (!fs.existsSync(dir)) {
    console.error(`Bundle directory not found: ${dir}`);
    process.exit(1);
  }

  if (!fs.existsSync(budgetFile)) {
    console.error(`Budget file not found: ${budgetFile}`);
    process.exit(1);
  }

  const budget = JSON.parse(fs.readFileSync(budgetFile, 'utf8'));
  const report = createBundleSizeReport(dir, { topN: 5 });
  const evaluation = evaluateBundleBudget(report, budget);

  if (!evaluation.ok) {
    console.error('Bundle budget check failed:');
    for (const failure of evaluation.failures) {
      console.error(`- ${failure}`);
    }
    process.exit(1);
  }

  console.log('Bundle budget check passed.');
  console.log(`Total: ${formatBytes(report.totalBytes)} (budget ${formatBytes(budget.totalMaxBytes)})`);
  if (report.topFiles[0]) {
    console.log(
      `Largest: ${formatBytes(report.topFiles[0].bytes)} (budget ${formatBytes(budget.largestAssetMaxBytes)})`
    );
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
