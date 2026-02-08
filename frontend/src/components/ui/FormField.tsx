import React, { useId } from 'react';

interface FormFieldProps {
  /** Label text displayed above the input */
  label: string;
  /** Optional helper text displayed below the input */
  helperText?: string;
  /** Error message - displays in red if present */
  error?: string | null;
  /** Whether the field has been touched/interacted with */
  touched?: boolean;
  /** Whether the field is required */
  required?: boolean;
  /** The input element(s) to render */
  children: React.ReactElement<React.InputHTMLAttributes<HTMLInputElement>>;
  /** Additional CSS classes for the container */
  className?: string;
  /** Show error only when touched (default: true) */
  showErrorOnlyWhenTouched?: boolean;
}

/**
 * Accessible form field wrapper that properly links labels to inputs.
 * Supports real-time validation with touched state for better UX.
 *
 * @example
 * // Basic usage
 * <FormField label="Email" required helperText="We'll never share your email">
 *   <input type="email" className="..." />
 * </FormField>
 *
 * @example
 * // With validation hook
 * <FormField
 *   label="Email"
 *   required
 *   error={errors.email}
 *   touched={touched.email}
 * >
 *   <input {...getFieldProps('email')} className="..." />
 * </FormField>
 */
const FormField: React.FC<FormFieldProps> = ({
  label,
  helperText,
  error,
  touched = true,
  required = false,
  children,
  className = '',
  showErrorOnlyWhenTouched = true,
}) => {
  const id = useId();
  const helperId = `${id}-helper`;
  const errorId = `${id}-error`;

  // Determine if we should show the error
  const showError = error && (showErrorOnlyWhenTouched ? touched : true);

  // Get the existing className from the child input
  const childProps = children.props as { className?: string };
  const existingClassName = childProps.className || '';

  // Build error-aware input styles
  const inputClassName = showError
    ? existingClassName.replace(
        /border-(?:gray|slate)-\d{3}/g,
        'border-red-500'
      ).replace(
        /focus:(?:ring|border)-(?:cyan|blue)-\d{3}/g,
        'focus:ring-red-500 focus:border-red-500'
      ) + ' border-red-500'
    : existingClassName;

  // Clone the child input to add accessibility attributes and error styling
  const enhancedChild = React.cloneElement(children, {
    id,
    className: inputClassName,
    'aria-describedby': showError ? errorId : helperText ? helperId : undefined,
    'aria-invalid': showError ? 'true' : undefined,
    'aria-required': required ? 'true' : undefined,
  });

  return (
    <div className={className}>
      <label
        htmlFor={id}
        className={`block text-sm font-medium mb-2 transition-colors ${
          showError ? 'text-red-400' : 'text-gray-300'
        }`}
      >
        {label}
        {required && (
          <span className="text-red-400 ml-1" aria-hidden="true">*</span>
        )}
      </label>

      {enhancedChild}

      {showError && (
        <p
          id={errorId}
          className="mt-1.5 text-sm text-red-400 flex items-center gap-1 animate-in fade-in slide-in-from-top-1 duration-200"
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

      {helperText && !showError && (
        <p id={helperId} className="mt-1.5 text-xs text-gray-400">
          {helperText}
        </p>
      )}
    </div>
  );
};

export default FormField;
