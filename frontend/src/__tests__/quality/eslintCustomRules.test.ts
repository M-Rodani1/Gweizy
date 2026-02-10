import { RuleTester } from 'eslint';
import tsparser from '@typescript-eslint/parser';
import requireRelOnTargetBlank from '../../../eslint-rules/require-rel-on-target-blank.js';

const ruleTester = new RuleTester({
  languageOptions: {
    parser: tsparser,
    parserOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      ecmaFeatures: { jsx: true }
    }
  }
});

ruleTester.run('require-rel-on-target-blank', requireRelOnTargetBlank, {
  valid: [
    {
      code: '<a href="https://example.com" target="_blank" rel="noopener noreferrer">Link</a>'
    },
    {
      code: '<a href="https://example.com" target="_blank" rel="noreferrer">Link</a>'
    },
    {
      code: '<a href="/internal">Internal</a>'
    }
  ],
  invalid: [
    {
      code: '<a href="https://example.com" target="_blank">Missing rel</a>',
      errors: [{ messageId: 'missingRel' }]
    },
    {
      code: '<a href="https://example.com" target="_blank" rel="nofollow">Bad rel</a>',
      errors: [{ messageId: 'missingRel' }]
    }
  ]
});
