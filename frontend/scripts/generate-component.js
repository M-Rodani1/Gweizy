#!/usr/bin/env node
/**
 * Component Generator Script
 *
 * Generates new React components with consistent structure.
 *
 * Usage:
 *   node scripts/generate-component.js ComponentName [--dir=path] [--type=ui|feature]
 *
 * Examples:
 *   node scripts/generate-component.js Button --dir=ui
 *   node scripts/generate-component.js GasChart --type=feature
 *   npm run generate:component -- MyComponent --dir=cards
 */

const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
const componentName = args.find(arg => !arg.startsWith('--'));

if (!componentName) {
  console.error('Error: Component name is required');
  console.log('Usage: node scripts/generate-component.js ComponentName [--dir=path] [--type=ui|feature]');
  process.exit(1);
}

// Validate component name
if (!/^[A-Z][a-zA-Z0-9]*$/.test(componentName)) {
  console.error('Error: Component name must start with uppercase letter and contain only alphanumeric characters');
  process.exit(1);
}

// Parse options
const getOption = (name, defaultValue) => {
  const arg = args.find(a => a.startsWith(`--${name}=`));
  return arg ? arg.split('=')[1] : defaultValue;
};

const hasFlag = (name) => args.includes(`--${name}`);

const componentDir = getOption('dir', 'ui');
const componentType = getOption('type', 'ui');
const withTest = !hasFlag('no-test');
const withStory = !hasFlag('no-story');

// Determine paths
const baseDir = path.join(__dirname, '..', 'src', 'components', componentDir);
const componentPath = path.join(baseDir, `${componentName}.tsx`);
const testPath = path.join(__dirname, '..', 'src', '__tests__', 'components', `${componentName}.test.tsx`);
const storyPath = path.join(__dirname, '..', 'src', 'stories', `${componentName}.stories.tsx`);

// Check if component already exists
if (fs.existsSync(componentPath)) {
  console.error(`Error: Component ${componentName} already exists at ${componentPath}`);
  process.exit(1);
}

// Component template
const componentTemplate = `/**
 * ${componentName} Component
 *
 * @module components/${componentDir}/${componentName}
 */

import React from 'react';

export interface ${componentName}Props {
  /** Optional className for styling */
  className?: string;
  /** Children elements */
  children?: React.ReactNode;
}

/**
 * ${componentName} component.
 *
 * @example
 * \`\`\`tsx
 * <${componentName}>
 *   Content here
 * </${componentName}>
 * \`\`\`
 */
export const ${componentName}: React.FC<${componentName}Props> = ({
  className = '',
  children,
}) => {
  return (
    <div className={\`\${className}\`.trim()}>
      {children}
    </div>
  );
};

export default ${componentName};
`;

// Test template
const testTemplate = `/**
 * Tests for ${componentName} component
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ${componentName} } from '../../components/${componentDir}/${componentName}';

describe('${componentName}', () => {
  it('renders children correctly', () => {
    render(<${componentName}>Test content</${componentName}>);
    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <${componentName} className="custom-class">Content</${componentName}>
    );
    expect(container.firstChild).toHaveClass('custom-class');
  });

  it('renders without children', () => {
    const { container } = render(<${componentName} />);
    expect(container.firstChild).toBeInTheDocument();
  });
});
`;

// Storybook story template
const storyTemplate = `/**
 * Storybook stories for ${componentName}
 */

import type { Meta, StoryObj } from '@storybook/react';
import { ${componentName} } from '../components/${componentDir}/${componentName}';

const meta: Meta<typeof ${componentName}> = {
  title: 'Components/${componentDir === 'ui' ? 'UI' : componentDir}/${componentName}',
  component: ${componentName},
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    className: {
      control: 'text',
      description: 'Additional CSS classes',
    },
    children: {
      control: 'text',
      description: 'Content to render inside the component',
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    children: 'Default ${componentName}',
  },
};

export const WithClassName: Story = {
  args: {
    children: '${componentName} with custom styling',
    className: 'p-4 bg-gray-800 rounded-lg',
  },
};

export const Empty: Story = {
  args: {},
};
`;

// Create directories if they don't exist
const ensureDir = (dirPath) => {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
};

// Write files
try {
  // Create component
  ensureDir(baseDir);
  fs.writeFileSync(componentPath, componentTemplate);
  console.log(`✓ Created component: ${componentPath}`);

  // Create test
  if (withTest) {
    ensureDir(path.dirname(testPath));
    fs.writeFileSync(testPath, testTemplate);
    console.log(`✓ Created test: ${testPath}`);
  }

  // Create story
  if (withStory) {
    ensureDir(path.dirname(storyPath));
    fs.writeFileSync(storyPath, storyTemplate);
    console.log(`✓ Created story: ${storyPath}`);
  }

  console.log(`\n✨ Successfully generated ${componentName} component!`);
  console.log('\nNext steps:');
  console.log(`  1. Update the component implementation in ${componentPath}`);
  console.log(`  2. Add exports to src/components/${componentDir}/index.ts (if exists)`);
  if (withTest) {
    console.log(`  3. Run tests: npm test`);
  }
  if (withStory) {
    console.log(`  4. View in Storybook: npm run storybook`);
  }

} catch (error) {
  console.error('Error generating component:', error.message);
  process.exit(1);
}
