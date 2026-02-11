import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useFormValidation, required, email } from '../../hooks/useFormValidation';

function TestForm() {
  const {
    errors,
    touched,
    getFieldProps,
    validateAll,
    setTouched,
  } = useFormValidation({
    email: {
      initialValue: '',
      rules: [required('Email required'), email('Email invalid')],
    },
  });

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (validateAll()) {
      setTouched('email', true);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="email">Email</label>
      <input id="email" type="email" {...getFieldProps('email')} />
      {touched.email && errors.email && (
        <div role="alert">{errors.email}</div>
      )}
      <button type="submit">Submit</button>
    </form>
  );
}

describe('form validation end-to-end', () => {
  it('shows required and format errors, then accepts valid input', () => {
    render(<TestForm />);

    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Email required');

    const input = screen.getByLabelText('Email');
    fireEvent.change(input, { target: { value: 'invalid' } });
    fireEvent.blur(input);
    expect(screen.getByRole('alert')).toHaveTextContent('Email invalid');

    fireEvent.change(input, { target: { value: 'valid@example.com' } });
    fireEvent.blur(input);
    fireEvent.click(screen.getByRole('button', { name: 'Submit' }));

    expect(screen.queryByRole('alert')).toBeNull();
  });
});
