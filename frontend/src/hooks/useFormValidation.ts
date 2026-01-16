/**
 * Custom hook for real-time form validation.
 *
 * Provides field-level validation with support for common rules,
 * touched state tracking, and real-time error display.
 *
 * @module hooks/useFormValidation
 */

import { useState, useCallback, useMemo } from 'react';

/**
 * Validation rule definition.
 */
export interface ValidationRule {
  /** Validation function that returns error message or null/undefined if valid */
  validate: (value: string, allValues?: Record<string, string>) => string | null | undefined;
  /** When to run this validation: 'change' (real-time), 'blur', or 'submit' */
  validateOn?: 'change' | 'blur' | 'submit';
}

/**
 * Field configuration with validation rules.
 */
export interface FieldConfig {
  /** Initial value for the field */
  initialValue?: string;
  /** Array of validation rules to apply */
  rules?: ValidationRule[];
}

/**
 * Schema defining all form fields and their configurations.
 */
export type ValidationSchema<T extends string> = Record<T, FieldConfig>;

/**
 * Field state returned by the hook.
 */
export interface FieldState {
  value: string;
  error: string | null;
  touched: boolean;
  isValid: boolean;
}

/**
 * Return type of useFormValidation hook.
 */
export interface UseFormValidationReturn<T extends string> {
  /** Current values for all fields */
  values: Record<T, string>;
  /** Current errors for all fields (null if no error) */
  errors: Record<T, string | null>;
  /** Touched state for all fields */
  touched: Record<T, boolean>;
  /** Whether the entire form is valid */
  isValid: boolean;
  /** Whether any field has been modified */
  isDirty: boolean;
  /** Get all state for a specific field */
  getFieldState: (field: T) => FieldState;
  /** Get props to spread on an input element */
  getFieldProps: (field: T) => {
    value: string;
    onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => void;
    onBlur: () => void;
    'aria-invalid': boolean;
    'aria-describedby': string | undefined;
  };
  /** Set a field's value programmatically */
  setValue: (field: T, value: string) => void;
  /** Set multiple values at once */
  setValues: (values: Partial<Record<T, string>>) => void;
  /** Mark a field as touched */
  setTouched: (field: T, isTouched?: boolean) => void;
  /** Set a custom error for a field */
  setError: (field: T, error: string | null) => void;
  /** Validate a single field and return whether it's valid */
  validateField: (field: T) => boolean;
  /** Validate all fields and return whether form is valid */
  validateAll: () => boolean;
  /** Reset form to initial values */
  reset: () => void;
  /** Reset a single field */
  resetField: (field: T) => void;
}

// ============================================================================
// Built-in Validators
// ============================================================================

/**
 * Creates a required field validator.
 * @param message - Custom error message
 */
export const required = (message = 'This field is required'): ValidationRule => ({
  validate: (value) => (!value || !value.trim() ? message : null),
  validateOn: 'change',
});

/**
 * Creates an email validator.
 * @param message - Custom error message
 */
export const email = (message = 'Please enter a valid email address'): ValidationRule => ({
  validate: (value) => {
    if (!value) return null; // Use required() for empty check
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(value) ? null : message;
  },
  validateOn: 'blur',
});

/**
 * Creates an Ethereum address validator.
 * @param message - Custom error message
 */
export const ethereumAddress = (message = 'Please enter a valid Ethereum address'): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    const ethRegex = /^0x[a-fA-F0-9]{40}$/;
    return ethRegex.test(value) ? null : message;
  },
  validateOn: 'blur',
});

/**
 * Creates a minimum value validator for numbers.
 * @param min - Minimum allowed value
 * @param message - Custom error message (use {min} as placeholder)
 */
export const minValue = (min: number, message?: string): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    const num = parseFloat(value);
    if (isNaN(num)) return 'Please enter a valid number';
    return num >= min ? null : (message || `Value must be at least ${min}`);
  },
  validateOn: 'change',
});

/**
 * Creates a maximum value validator for numbers.
 * @param max - Maximum allowed value
 * @param message - Custom error message (use {max} as placeholder)
 */
export const maxValue = (max: number, message?: string): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    const num = parseFloat(value);
    if (isNaN(num)) return 'Please enter a valid number';
    return num <= max ? null : (message || `Value must be at most ${max}`);
  },
  validateOn: 'change',
});

/**
 * Creates a minimum length validator.
 * @param length - Minimum required length
 * @param message - Custom error message
 */
export const minLength = (length: number, message?: string): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    return value.length >= length ? null : (message || `Must be at least ${length} characters`);
  },
  validateOn: 'change',
});

/**
 * Creates a maximum length validator.
 * @param length - Maximum allowed length
 * @param message - Custom error message
 */
export const maxLength = (length: number, message?: string): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    return value.length <= length ? null : (message || `Must be at most ${length} characters`);
  },
  validateOn: 'change',
});

/**
 * Creates a pattern validator using a regular expression.
 * @param pattern - RegExp to test against
 * @param message - Error message when pattern doesn't match
 */
export const pattern = (regex: RegExp, message: string): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    return regex.test(value) ? null : message;
  },
  validateOn: 'blur',
});

/**
 * Creates a positive number validator.
 * @param message - Custom error message
 */
export const positiveNumber = (message = 'Please enter a positive number'): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    const num = parseFloat(value);
    if (isNaN(num)) return 'Please enter a valid number';
    return num > 0 ? null : message;
  },
  validateOn: 'change',
});

/**
 * Creates a URL validator.
 * @param message - Custom error message
 */
export const url = (message = 'Please enter a valid URL'): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    try {
      new URL(value);
      return null;
    } catch {
      return message;
    }
  },
  validateOn: 'blur',
});

/**
 * Creates a Discord webhook URL validator.
 * @param message - Custom error message
 */
export const discordWebhook = (message = 'Please enter a valid Discord webhook URL'): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    const discordRegex = /^https:\/\/discord\.com\/api\/webhooks\/\d+\/[\w-]+$/;
    return discordRegex.test(value) ? null : message;
  },
  validateOn: 'blur',
});

/**
 * Creates a numeric-only validator (for Telegram chat IDs, etc).
 * @param message - Custom error message
 */
export const numericOnly = (message = 'Please enter numbers only'): ValidationRule => ({
  validate: (value) => {
    if (!value) return null;
    return /^-?\d+$/.test(value) ? null : message;
  },
  validateOn: 'change',
});

// ============================================================================
// Main Hook
// ============================================================================

/**
 * Hook for managing form validation with real-time feedback.
 *
 * @template T - Union type of field names
 * @param schema - Validation schema defining fields and their rules
 * @returns Form validation state and helpers
 *
 * @example
 * ```tsx
 * const {
 *   values,
 *   errors,
 *   touched,
 *   isValid,
 *   getFieldProps,
 *   validateAll
 * } = useFormValidation({
 *   email: {
 *     initialValue: '',
 *     rules: [required(), email()]
 *   },
 *   amount: {
 *     initialValue: '0',
 *     rules: [required(), positiveNumber()]
 *   }
 * });
 *
 * return (
 *   <form onSubmit={(e) => {
 *     e.preventDefault();
 *     if (validateAll()) {
 *       // Submit form
 *     }
 *   }}>
 *     <input {...getFieldProps('email')} />
 *     {touched.email && errors.email && (
 *       <span className="text-red-500">{errors.email}</span>
 *     )}
 *   </form>
 * );
 * ```
 */
export function useFormValidation<T extends string>(
  schema: ValidationSchema<T>
): UseFormValidationReturn<T> {
  const fieldNames = Object.keys(schema) as T[];

  // Initialize values from schema
  const initialValues = useMemo(() => {
    const values: Record<string, string> = {};
    fieldNames.forEach((field) => {
      values[field] = schema[field].initialValue || '';
    });
    return values as Record<T, string>;
  }, []);

  const [values, setValuesState] = useState<Record<T, string>>(initialValues);
  const [errors, setErrorsState] = useState<Record<T, string | null>>(() => {
    const initial: Record<string, string | null> = {};
    fieldNames.forEach((field) => {
      initial[field] = null;
    });
    return initial as Record<T, string | null>;
  });
  const [touched, setTouchedState] = useState<Record<T, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    fieldNames.forEach((field) => {
      initial[field] = false;
    });
    return initial as Record<T, boolean>;
  });

  // Validate a single field
  const runValidation = useCallback(
    (field: T, value: string, trigger: 'change' | 'blur' | 'submit'): string | null => {
      const rules = schema[field]?.rules || [];

      for (const rule of rules) {
        const validateOn = rule.validateOn || 'submit';

        // Run validation based on trigger priority: submit > blur > change
        const shouldValidate =
          trigger === 'submit' ||
          (trigger === 'blur' && validateOn !== 'submit') ||
          (trigger === 'change' && validateOn === 'change');

        if (shouldValidate) {
          const error = rule.validate(value, values);
          if (error) {
            return error;
          }
        }
      }

      return null;
    },
    [schema, values]
  );

  // Set a single field value
  const setValue = useCallback(
    (field: T, value: string) => {
      setValuesState((prev) => ({ ...prev, [field]: value }));

      // Run change-time validation
      const error = runValidation(field, value, 'change');
      setErrorsState((prev) => ({ ...prev, [field]: error }));
    },
    [runValidation]
  );

  // Set multiple values at once
  const setValues = useCallback(
    (newValues: Partial<Record<T, string>>) => {
      setValuesState((prev) => ({ ...prev, ...newValues }));

      // Validate each changed field
      const newErrors: Partial<Record<T, string | null>> = {};
      (Object.keys(newValues) as T[]).forEach((field) => {
        newErrors[field] = runValidation(field, newValues[field]!, 'change');
      });
      setErrorsState((prev) => ({ ...prev, ...newErrors }));
    },
    [runValidation]
  );

  // Mark field as touched and run blur validation
  const setTouched = useCallback(
    (field: T, isTouched = true) => {
      setTouchedState((prev) => ({ ...prev, [field]: isTouched }));

      if (isTouched) {
        // Run blur-time validation
        const error = runValidation(field, values[field], 'blur');
        setErrorsState((prev) => ({ ...prev, [field]: error }));
      }
    },
    [runValidation, values]
  );

  // Set a custom error
  const setError = useCallback((field: T, error: string | null) => {
    setErrorsState((prev) => ({ ...prev, [field]: error }));
  }, []);

  // Validate a single field (for submit)
  const validateField = useCallback(
    (field: T): boolean => {
      const error = runValidation(field, values[field], 'submit');
      setErrorsState((prev) => ({ ...prev, [field]: error }));
      setTouchedState((prev) => ({ ...prev, [field]: true }));
      return error === null;
    },
    [runValidation, values]
  );

  // Validate all fields
  const validateAll = useCallback((): boolean => {
    let isValid = true;
    const newErrors: Record<string, string | null> = {};
    const newTouched: Record<string, boolean> = {};

    fieldNames.forEach((field) => {
      const error = runValidation(field, values[field], 'submit');
      newErrors[field] = error;
      newTouched[field] = true;
      if (error) {
        isValid = false;
      }
    });

    setErrorsState(newErrors as Record<T, string | null>);
    setTouchedState(newTouched as Record<T, boolean>);
    return isValid;
  }, [fieldNames, runValidation, values]);

  // Reset entire form
  const reset = useCallback(() => {
    setValuesState(initialValues);
    setErrorsState(() => {
      const initial: Record<string, string | null> = {};
      fieldNames.forEach((field) => {
        initial[field] = null;
      });
      return initial as Record<T, string | null>;
    });
    setTouchedState(() => {
      const initial: Record<string, boolean> = {};
      fieldNames.forEach((field) => {
        initial[field] = false;
      });
      return initial as Record<T, boolean>;
    });
  }, [initialValues, fieldNames]);

  // Reset a single field
  const resetField = useCallback(
    (field: T) => {
      setValuesState((prev) => ({ ...prev, [field]: initialValues[field] }));
      setErrorsState((prev) => ({ ...prev, [field]: null }));
      setTouchedState((prev) => ({ ...prev, [field]: false }));
    },
    [initialValues]
  );

  // Get state for a single field
  const getFieldState = useCallback(
    (field: T): FieldState => ({
      value: values[field],
      error: errors[field],
      touched: touched[field],
      isValid: errors[field] === null,
    }),
    [values, errors, touched]
  );

  // Get props to spread on input elements
  const getFieldProps = useCallback(
    (field: T) => ({
      value: values[field],
      onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
        setValue(field, e.target.value);
      },
      onBlur: () => {
        setTouched(field, true);
      },
      'aria-invalid': touched[field] && errors[field] !== null,
      'aria-describedby': errors[field] ? `${field}-error` : undefined,
    }),
    [values, errors, touched, setValue, setTouched]
  );

  // Computed properties
  const isValid = useMemo(
    () => fieldNames.every((field) => errors[field] === null),
    [errors, fieldNames]
  );

  const isDirty = useMemo(
    () => fieldNames.some((field) => values[field] !== initialValues[field]),
    [values, initialValues, fieldNames]
  );

  return {
    values,
    errors,
    touched,
    isValid,
    isDirty,
    getFieldState,
    getFieldProps,
    setValue,
    setValues,
    setTouched,
    setError,
    validateField,
    validateAll,
    reset,
    resetField,
  };
}

export default useFormValidation;
