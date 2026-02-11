import { readdirSync, statSync, readFileSync } from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(process.cwd(), 'src');

const isSourceFile = (file) => file.endsWith('.ts') || file.endsWith('.tsx');

const walk = (dir, files = []) => {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry);
    const stats = statSync(full);
    if (stats.isDirectory()) {
      if (entry === 'node_modules' || entry === 'dist') continue;
      walk(full, files);
      continue;
    }
    if (isSourceFile(entry)) files.push(full);
  }
  return files;
};

const resolveImport = (from, spec) => {
  if (!spec.startsWith('.')) return null;
  const base = path.resolve(path.dirname(from), spec);
  const candidates = [
    `${base}.ts`,
    `${base}.tsx`,
    path.join(base, 'index.ts'),
    path.join(base, 'index.tsx')
  ];
  return candidates.find((file) => statExists(file)) || null;
};

const statExists = (file) => {
  try {
    return statSync(file).isFile();
  } catch {
    return false;
  }
};

const extractImports = (file) => {
  const contents = readFileSync(file, 'utf8');
  const matches = contents.matchAll(/from\s+['"]([^'"]+)['"]/g);
  const imports = [];
  for (const match of matches) {
    imports.push(match[1]);
  }
  return imports;
};

const buildGraph = (files) => {
  const graph = new Map();
  files.forEach((file) => {
    const deps = extractImports(file)
      .map((spec) => resolveImport(file, spec))
      .filter(Boolean);
    graph.set(file, deps);
  });
  return graph;
};

const findCycles = (graph) => {
  const visited = new Set();
  const stack = new Set();
  const cycles = [];

  const visit = (node, pathStack) => {
    if (stack.has(node)) {
      const cycleStart = pathStack.indexOf(node);
      cycles.push(pathStack.slice(cycleStart).concat(node));
      return;
    }
    if (visited.has(node)) return;
    visited.add(node);
    stack.add(node);
    const deps = graph.get(node) || [];
    deps.forEach((dep) => visit(dep, pathStack.concat(dep)));
    stack.delete(node);
  };

  graph.forEach((_deps, node) => {
    if (!visited.has(node)) visit(node, [node]);
  });

  return cycles;
};

const main = () => {
  const files = walk(ROOT);
  const graph = buildGraph(files);
  const cycles = findCycles(graph);

  if (cycles.length) {
    console.error('Circular dependencies detected:');
    cycles.forEach((cycle) => {
      console.error(cycle.map((file) => path.relative(ROOT, file)).join(' -> '));
    });
    process.exit(1);
  }

  console.log('No circular dependencies detected.');
};

main();
