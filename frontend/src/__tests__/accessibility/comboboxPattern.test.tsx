import { fireEvent, render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import Combobox, { ComboboxOption } from '../../components/ui/Combobox';

const options: ComboboxOption[] = [
  { id: 'eth', label: 'Ethereum', value: 'eth' },
  { id: 'base', label: 'Base', value: 'base' },
  { id: 'arb', label: 'Arbitrum', value: 'arb' },
];

describe('Combobox accessibility pattern', () => {
  it('renders combobox roles and listbox options', () => {
    render(
      <Combobox
        label="Select network"
        options={options}
        onSelect={vi.fn()}
      />
    );

    const input = screen.getByRole('combobox', { name: 'Select network' });
    fireEvent.change(input, { target: { value: 'a' } });

    const listbox = screen.getByRole('listbox');
    expect(listbox).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Base' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Arbitrum' })).toBeInTheDocument();
  });

  it('updates active descendant with keyboard navigation', () => {
    render(
      <Combobox
        label="Select network"
        options={options}
        onSelect={vi.fn()}
      />
    );

    const input = screen.getByRole('combobox', { name: 'Select network' });
    fireEvent.change(input, { target: { value: '' } });
    fireEvent.keyDown(input, { key: 'ArrowDown' });

    expect(input).toHaveAttribute('aria-activedescendant');
  });

  it('selects an option with Enter', () => {
    const onSelect = vi.fn();
    render(
      <Combobox
        label="Select network"
        options={options}
        onSelect={onSelect}
      />
    );

    const input = screen.getByRole('combobox', { name: 'Select network' });
    fireEvent.change(input, { target: { value: 'eth' } });
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(onSelect).toHaveBeenCalledWith(options[0]);
  });
});
