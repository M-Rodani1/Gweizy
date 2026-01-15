import React, { useId } from 'react';

interface FormFieldProps {
  /** Label text displayed above the input */
  label: string;
  /** Optional helper text displayed below the input */
  helperText?: string;
  /** Error message - displays in red if present */
  error?: string;
  /** Whether the field is required */
  required?: boolean;
  /** The input element(s) to render */
  children: React.ReactElement;
  /** Additional CSS classes for the container */
  className?: string;
}

/**
 * Accessible form field wrapper that properly links labels to inputs.
 *
 * @example
 * <FormField label="Email" required helperText="We'll never share your email">
 *   <input type="email" className="..." />
 * </FormField>
 */
const FormField: React.FC<FormFieldProps> = ({
  label,
  helperText,
  error,
  required = false,
  children,
  className = '',
}) => {
  const id = useId();
  const helperId = `${id}-helper`;
  const errorId = `${id}-error`;

  // Clone the child input to add accessibility attributes
  const enhancedChild = React.cloneElement(children, {
    id,
    'aria-describedby': error ? errorId : helperText ? helperId : undefined,
    'aria-invalid': error ? 'true' : undefined,
    'aria-required': required ? 'true' : undefined,
  });

  return (
    <div className={className}>
      <label
        htmlFor={id}
        className="block text-sm font-medium text-gray-300 mb-2"
      >
        {label}
        {required && (
          <span className="text-red-400 ml-1" aria-hidden="true">*</span>
        )}
      </label>

      {enhancedChild}

      {error && (
        <p
          id={errorId}
          className="mt-1.5 text-sm text-red-400 flex items-center gap-1"
          role="alert"
        >
          <svg
            className="w-4 h-4 flex-shrink-0"
            fill="currentColor"
            viewBox="0 0 20 20"
            aria-hidden="true"
          >
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
          {error}
        </p>
      )}

      {helperText && !error && (
        <p id={helperId} className="mt-1.5 text-xs text-gray-400">
          {helperText}
        </p>
      )}
    </div>
  );
};

export default FormField;
