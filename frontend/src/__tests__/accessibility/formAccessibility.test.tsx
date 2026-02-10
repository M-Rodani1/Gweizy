import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import FormField from '../../components/ui/FormField';

describe('form accessibility', () => {
  it('links labels to inputs and sets helper text', () => {
    render(
      <FormField label="Email" helperText="We will not spam you">
        <input type="email" />
      </FormField>
    );

    const input = screen.getByLabelText('Email');
    expect(input).toHaveAttribute('aria-describedby');
  });

  it('marks required fields and exposes errors', () => {
    render(
      <FormField label="Password" required error="Required">
        <input type="password" />
      </FormField>
    );

    const input = screen.getByLabelText('Password', { exact: false });
    expect(input).toHaveAttribute('aria-required', 'true');
    expect(input).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent('Required');
  });
});
