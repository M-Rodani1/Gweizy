/**
 * Tests for useFormValidation hook
 *
 * Comprehensive tests for the form validation hook including:
 * - Built-in validators
 * - Real-time validation
 * - Touch state management
 * - Form submission validation
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import {
  useFormValidation,
  required,
  email,
  ethereumAddress,
  positiveNumber,
  minValue,
  maxValue,
  minLength,
  maxLength,
  pattern,
  url,
  discordWebhook,
  numericOnly,
} from '../../hooks/useFormValidation';

describe('useFormValidation', () => {
  describe('Initial State', () => {
    it('should initialize with default values from schema', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: 'John' },
          email: { initialValue: 'john@example.com' },
        })
      );

      expect(result.current.values.name).toBe('John');
      expect(result.current.values.email).toBe('john@example.com');
    });

    it('should initialize with empty strings when no initial value provided', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: {},
          email: {},
        })
      );

      expect(result.current.values.name).toBe('');
      expect(result.current.values.email).toBe('');
    });

    it('should start with no errors', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { rules: [required()] },
        })
      );

      expect(result.current.errors.name).toBeNull();
    });

    it('should start with untouched fields', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: {},
          email: {},
        })
      );

      expect(result.current.touched.name).toBe(false);
      expect(result.current.touched.email).toBe(false);
    });

    it('should report isValid as true when no errors', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: 'John' },
        })
      );

      expect(result.current.isValid).toBe(true);
    });

    it('should report isDirty as false initially', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: 'John' },
        })
      );

      expect(result.current.isDirty).toBe(false);
    });
  });

  describe('setValue', () => {
    it('should update field value', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
        })
      );

      act(() => {
        result.current.setValue('name', 'Jane');
      });

      expect(result.current.values.name).toBe('Jane');
    });

    it('should run change-time validation', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          amount: {
            initialValue: '',
            rules: [required('Amount is required')],
          },
        })
      );

      act(() => {
        result.current.setValue('amount', '');
      });

      expect(result.current.errors.amount).toBe('Amount is required');
    });

    it('should clear error when valid value is entered', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: {
            initialValue: '',
            rules: [required()],
          },
        })
      );

      act(() => {
        result.current.setValue('name', '');
      });

      expect(result.current.errors.name).toBeTruthy();

      act(() => {
        result.current.setValue('name', 'John');
      });

      expect(result.current.errors.name).toBeNull();
    });

    it('should set isDirty to true when value changes', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
        })
      );

      act(() => {
        result.current.setValue('name', 'Jane');
      });

      expect(result.current.isDirty).toBe(true);
    });
  });

  describe('setValues', () => {
    it('should update multiple values at once', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
          email: { initialValue: '' },
        })
      );

      act(() => {
        result.current.setValues({
          name: 'John',
          email: 'john@example.com',
        });
      });

      expect(result.current.values.name).toBe('John');
      expect(result.current.values.email).toBe('john@example.com');
    });

    it('should validate all changed fields', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '', rules: [required()] },
          email: { initialValue: '', rules: [required()] },
        })
      );

      act(() => {
        result.current.setValues({
          name: '',
          email: 'valid@email.com',
        });
      });

      expect(result.current.errors.name).toBeTruthy();
      expect(result.current.errors.email).toBeNull();
    });
  });

  describe('setTouched', () => {
    it('should mark field as touched', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
        })
      );

      act(() => {
        result.current.setTouched('name', true);
      });

      expect(result.current.touched.name).toBe(true);
    });

    it('should run blur-time validation when touched', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          email: {
            initialValue: 'invalid',
            rules: [email()],
          },
        })
      );

      expect(result.current.errors.email).toBeNull();

      act(() => {
        result.current.setTouched('email', true);
      });

      expect(result.current.errors.email).toBeTruthy();
    });
  });

  describe('setError', () => {
    it('should set custom error', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
        })
      );

      act(() => {
        result.current.setError('name', 'Custom error message');
      });

      expect(result.current.errors.name).toBe('Custom error message');
    });

    it('should clear error when set to null', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
        })
      );

      act(() => {
        result.current.setError('name', 'Error');
      });

      act(() => {
        result.current.setError('name', null);
      });

      expect(result.current.errors.name).toBeNull();
    });
  });

  describe('validateField', () => {
    it('should validate single field with all rules', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          email: {
            initialValue: 'invalid',
            rules: [required(), email()],
          },
        })
      );

      let isValid: boolean;
      act(() => {
        isValid = result.current.validateField('email');
      });

      expect(isValid!).toBe(false);
      expect(result.current.errors.email).toBeTruthy();
    });

    it('should mark field as touched', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
        })
      );

      act(() => {
        result.current.validateField('name');
      });

      expect(result.current.touched.name).toBe(true);
    });

    it('should return true when field is valid', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          email: {
            initialValue: 'valid@example.com',
            rules: [required(), email()],
          },
        })
      );

      let isValid: boolean;
      act(() => {
        isValid = result.current.validateField('email');
      });

      expect(isValid!).toBe(true);
    });
  });

  describe('validateAll', () => {
    it('should validate all fields', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '', rules: [required()] },
          email: { initialValue: 'invalid', rules: [email()] },
        })
      );

      let isValid: boolean;
      act(() => {
        isValid = result.current.validateAll();
      });

      expect(isValid!).toBe(false);
      expect(result.current.errors.name).toBeTruthy();
      expect(result.current.errors.email).toBeTruthy();
    });

    it('should mark all fields as touched', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
          email: { initialValue: '' },
        })
      );

      act(() => {
        result.current.validateAll();
      });

      expect(result.current.touched.name).toBe(true);
      expect(result.current.touched.email).toBe(true);
    });

    it('should return true when all fields are valid', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: 'John', rules: [required()] },
          email: { initialValue: 'john@example.com', rules: [email()] },
        })
      );

      let isValid: boolean;
      act(() => {
        isValid = result.current.validateAll();
      });

      expect(isValid!).toBe(true);
    });
  });

  describe('reset', () => {
    it('should reset values to initial values', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: 'John' },
        })
      );

      act(() => {
        result.current.setValue('name', 'Jane');
      });

      act(() => {
        result.current.reset();
      });

      expect(result.current.values.name).toBe('John');
    });

    it('should clear all errors', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '', rules: [required()] },
        })
      );

      act(() => {
        result.current.validateAll();
      });

      act(() => {
        result.current.reset();
      });

      expect(result.current.errors.name).toBeNull();
    });

    it('should reset touched state', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '' },
        })
      );

      act(() => {
        result.current.setTouched('name', true);
      });

      act(() => {
        result.current.reset();
      });

      expect(result.current.touched.name).toBe(false);
    });
  });

  describe('resetField', () => {
    it('should reset single field to initial value', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: 'John' },
          email: { initialValue: 'john@example.com' },
        })
      );

      act(() => {
        result.current.setValue('name', 'Jane');
        result.current.setValue('email', 'jane@example.com');
      });

      act(() => {
        result.current.resetField('name');
      });

      expect(result.current.values.name).toBe('John');
      expect(result.current.values.email).toBe('jane@example.com');
    });
  });

  describe('getFieldState', () => {
    it('should return complete field state', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: 'John', rules: [required()] },
        })
      );

      const fieldState = result.current.getFieldState('name');

      expect(fieldState).toEqual({
        value: 'John',
        error: null,
        touched: false,
        isValid: true,
      });
    });
  });

  describe('getFieldProps', () => {
    it('should return props for input binding', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: 'John' },
        })
      );

      const props = result.current.getFieldProps('name');

      expect(props.value).toBe('John');
      expect(typeof props.onChange).toBe('function');
      expect(typeof props.onBlur).toBe('function');
      expect(props['aria-invalid']).toBe(false);
    });

    it('should set aria-invalid when field has error and is touched', () => {
      const { result } = renderHook(() =>
        useFormValidation({
          name: { initialValue: '', rules: [required()] },
        })
      );

      act(() => {
        result.current.setTouched('name', true);
        result.current.validateField('name');
      });

      const props = result.current.getFieldProps('name');
      expect(props['aria-invalid']).toBe(true);
    });
  });
});

describe('Built-in Validators', () => {
  describe('required', () => {
    it('should fail for empty string', () => {
      const rule = required();
      expect(rule.validate('')).toBeTruthy();
    });

    it('should fail for whitespace-only string', () => {
      const rule = required();
      expect(rule.validate('   ')).toBeTruthy();
    });

    it('should pass for non-empty string', () => {
      const rule = required();
      expect(rule.validate('value')).toBeNull();
    });

    it('should use custom message', () => {
      const rule = required('Custom required message');
      expect(rule.validate('')).toBe('Custom required message');
    });
  });

  describe('email', () => {
    it('should pass for valid email', () => {
      const rule = email();
      expect(rule.validate('test@example.com')).toBeNull();
    });

    it('should fail for invalid email', () => {
      const rule = email();
      expect(rule.validate('invalid-email')).toBeTruthy();
    });

    it('should pass for empty value (use required for empty check)', () => {
      const rule = email();
      expect(rule.validate('')).toBeNull();
    });

    it('should use custom message', () => {
      const rule = email('Invalid email format');
      expect(rule.validate('invalid')).toBe('Invalid email format');
    });
  });

  describe('ethereumAddress', () => {
    it('should pass for valid address', () => {
      const rule = ethereumAddress();
      expect(rule.validate('0x742d35Cc6634C0532925a3b844Bc9e7595f8fE00')).toBeNull();
    });

    it('should fail for address without 0x prefix', () => {
      const rule = ethereumAddress();
      expect(rule.validate('742d35Cc6634C0532925a3b844Bc9e7595f8fE00')).toBeTruthy();
    });

    it('should fail for short address', () => {
      const rule = ethereumAddress();
      expect(rule.validate('0x742d35Cc')).toBeTruthy();
    });

    it('should fail for address with invalid characters', () => {
      const rule = ethereumAddress();
      expect(rule.validate('0x742d35Cc6634C0532925a3b844Bc9e7595f8fGGG')).toBeTruthy();
    });
  });

  describe('positiveNumber', () => {
    it('should pass for positive number', () => {
      const rule = positiveNumber();
      expect(rule.validate('5')).toBeNull();
      expect(rule.validate('0.001')).toBeNull();
    });

    it('should fail for zero', () => {
      const rule = positiveNumber();
      expect(rule.validate('0')).toBeTruthy();
    });

    it('should fail for negative number', () => {
      const rule = positiveNumber();
      expect(rule.validate('-5')).toBeTruthy();
    });

    it('should fail for non-numeric value', () => {
      const rule = positiveNumber();
      expect(rule.validate('abc')).toBeTruthy();
    });
  });

  describe('minValue', () => {
    it('should pass for value above minimum', () => {
      const rule = minValue(10);
      expect(rule.validate('15')).toBeNull();
    });

    it('should pass for value equal to minimum', () => {
      const rule = minValue(10);
      expect(rule.validate('10')).toBeNull();
    });

    it('should fail for value below minimum', () => {
      const rule = minValue(10);
      expect(rule.validate('5')).toBeTruthy();
    });
  });

  describe('maxValue', () => {
    it('should pass for value below maximum', () => {
      const rule = maxValue(10);
      expect(rule.validate('5')).toBeNull();
    });

    it('should pass for value equal to maximum', () => {
      const rule = maxValue(10);
      expect(rule.validate('10')).toBeNull();
    });

    it('should fail for value above maximum', () => {
      const rule = maxValue(10);
      expect(rule.validate('15')).toBeTruthy();
    });
  });

  describe('minLength', () => {
    it('should pass for string meeting minimum length', () => {
      const rule = minLength(5);
      expect(rule.validate('hello')).toBeNull();
    });

    it('should fail for short string', () => {
      const rule = minLength(5);
      expect(rule.validate('hi')).toBeTruthy();
    });
  });

  describe('maxLength', () => {
    it('should pass for string within maximum length', () => {
      const rule = maxLength(5);
      expect(rule.validate('hello')).toBeNull();
    });

    it('should fail for long string', () => {
      const rule = maxLength(5);
      expect(rule.validate('hello world')).toBeTruthy();
    });
  });

  describe('pattern', () => {
    it('should pass when pattern matches', () => {
      const rule = pattern(/^[A-Z]+$/, 'Must be uppercase');
      expect(rule.validate('HELLO')).toBeNull();
    });

    it('should fail when pattern does not match', () => {
      const rule = pattern(/^[A-Z]+$/, 'Must be uppercase');
      expect(rule.validate('hello')).toBe('Must be uppercase');
    });
  });

  describe('url', () => {
    it('should pass for valid URL', () => {
      const rule = url();
      expect(rule.validate('https://example.com')).toBeNull();
    });

    it('should fail for invalid URL', () => {
      const rule = url();
      expect(rule.validate('not-a-url')).toBeTruthy();
    });
  });

  describe('discordWebhook', () => {
    it('should pass for valid Discord webhook', () => {
      const rule = discordWebhook();
      expect(rule.validate('https://discord.com/api/webhooks/123456789/abc-def_GHI')).toBeNull();
    });

    it('should fail for invalid Discord webhook', () => {
      const rule = discordWebhook();
      expect(rule.validate('https://example.com/webhook')).toBeTruthy();
    });
  });

  describe('numericOnly', () => {
    it('should pass for numeric string', () => {
      const rule = numericOnly();
      expect(rule.validate('12345')).toBeNull();
    });

    it('should pass for negative numeric string', () => {
      const rule = numericOnly();
      expect(rule.validate('-12345')).toBeNull();
    });

    it('should fail for string with letters', () => {
      const rule = numericOnly();
      expect(rule.validate('123abc')).toBeTruthy();
    });

    it('should fail for decimal numbers', () => {
      const rule = numericOnly();
      expect(rule.validate('123.45')).toBeTruthy();
    });
  });
});

describe('Validation Timing', () => {
  it('should run change-time validators on value change', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        amount: {
          initialValue: '',
          rules: [required()], // required has validateOn: 'change'
        },
      })
    );

    act(() => {
      result.current.setValue('amount', '');
    });

    expect(result.current.errors.amount).toBeTruthy();
  });

  it('should run blur-time validators on blur', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        email: {
          initialValue: 'invalid',
          rules: [email()], // email has validateOn: 'blur'
        },
      })
    );

    // Should not have error before blur
    expect(result.current.errors.email).toBeNull();

    act(() => {
      result.current.setTouched('email', true);
    });

    // Should have error after blur
    expect(result.current.errors.email).toBeTruthy();
  });

  it('should run all validators on submit (validateAll)', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        email: {
          initialValue: 'invalid',
          rules: [email()], // blur-time validator
        },
      })
    );

    // No error before validateAll
    expect(result.current.errors.email).toBeNull();

    act(() => {
      result.current.validateAll();
    });

    // Error after validateAll
    expect(result.current.errors.email).toBeTruthy();
  });
});
