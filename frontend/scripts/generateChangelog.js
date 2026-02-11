import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const changelogPath = path.resolve(process.cwd(), 'CHANGELOG.md');
const timestamp = new Date().toISOString().split('T')[0];
const header = `\n## ${timestamp}\n\n- Changelog entry placeholder.\n`;

let existing = '';
try {
  existing = readFileSync(changelogPath, 'utf8');
} catch {
  existing = '# Changelog\n';
}

writeFileSync(changelogPath, `${existing}${header}`, 'utf8');
console.log(`Changelog updated: ${changelogPath}`);
