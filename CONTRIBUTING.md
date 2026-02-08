# Contributing to Gweizy

Thank you for your interest in contributing to Gweizy! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Node.js 18+
- npm 9+
- Git

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/M-Rodani1/Gweizy.git
   cd Gweizy
   ```

2. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:3000 in your browser

## Project Structure

```
frontend/
├── src/
│   ├── api/          # API client functions
│   ├── components/   # React components
│   │   └── ui/       # Reusable UI primitives
│   ├── config/       # Configuration files
│   ├── contexts/     # React context providers
│   ├── hooks/        # Custom React hooks
│   ├── schemas/      # Zod validation schemas
│   ├── styles/       # CSS and design tokens
│   └── utils/        # Utility functions
├── pages/            # Route pages
└── types.ts          # TypeScript type definitions
```

## Code Style

### TypeScript

- Use strict typing; avoid `any` where possible
- Define interfaces for API responses and component props
- Use Zod schemas for runtime validation

### React

- Use functional components with hooks
- Wrap expensive components with `React.memo`
- Use `useMemo` and `useCallback` for performance optimization
- Follow the Rules of Hooks (enforced by ESLint)

### CSS

- Use CSS variables from `tokens.css` for colors
- Follow the existing Tailwind CSS patterns
- Use semantic class names from the design system

## Development Workflow

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run specific test file
npm test -- --run src/__tests__/hooks/useRecommendation.test.ts
```

### Linting

```bash
# Run ESLint
npm run lint

# Auto-fix issues
npm run lint:fix
```

### Building

```bash
# Production build
npm run build

# Preview production build
npm run preview
```

### Storybook

```bash
# Start Storybook
npm run storybook

# Build Storybook
npm run build-storybook
```

## Commit Guidelines

We follow conventional commit messages:

- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code refactoring
- `perf:` Performance improvement
- `test:` Adding tests
- `chore:` Maintenance tasks
- `docs:` Documentation updates
- `style:` Code style changes
- `types:` TypeScript type changes

Example:
```
feat: add gas price prediction chart
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure tests pass: `npm test`
4. Ensure build succeeds: `npm run build`
5. Submit a pull request with a clear description

## Design System

### Colors

Use CSS variables from `src/styles/tokens.css`:

- `--accent`: Primary brand color (cyan)
- `--success`, `--warning`, `--danger`: Semantic colors
- `--bg`, `--surface`: Background colors
- `--text`, `--text-secondary`: Text colors

### Components

Use components from `src/components/ui/`:

- `Button` - Standard button with variants
- `Card` - Container with title/subtitle
- `Badge` - Status indicators
- `Stat` - Metric display
- `Sparkline` - Inline charts

## API Integration

### Adding New API Endpoints

1. Add the endpoint to `src/config/api.ts`
2. Create typed response interface in `types.ts`
3. Add Zod schema in `src/schemas/api.ts`
4. Create fetch function in `src/api/gasApi.ts`

### Using the usePolling Hook

For data that needs periodic refresh:

```typescript
import { usePolling } from '../hooks/usePolling';

const { data, loading, error, refresh } = usePolling({
  fetcher: () => fetchData(),
  interval: 30000, // 30 seconds
});
```

## Questions?

Open an issue on GitHub if you have questions or need help.
