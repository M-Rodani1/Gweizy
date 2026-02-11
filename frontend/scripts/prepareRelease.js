import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const pkgPath = path.resolve(process.cwd(), 'package.json');
const pkg = JSON.parse(readFileSync(pkgPath, 'utf8'));
const version = pkg.version || '0.0.0';

const outputDir = path.resolve(process.cwd(), 'docs');
const outputFile = path.join(outputDir, 'release-notes.md');

mkdirSync(outputDir, { recursive: true });
const contents = `# Release ${version}\n\n- Automated release notes placeholder.\n`;
writeFileSync(outputFile, contents, 'utf8');

console.log(`Release notes prepared: ${outputFile}`);
