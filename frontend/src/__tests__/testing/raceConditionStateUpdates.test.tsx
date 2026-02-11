import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useFormValidation, required } from '../../hooks/useFormValidation';

describe('Race Condition State Updates', () => {
  it('preserves multiple setValue calls in the same tick', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        first: { initialValue: '', rules: [required('First required')] },
        second: { initialValue: '', rules: [required('Second required')] },
      })
    );

    act(() => {
      result.current.setValue('first', 'alpha');
      result.current.setValue('second', 'beta');
    });

    expect(result.current.values.first).toBe('alpha');
    expect(result.current.values.second).toBe('beta');
  });

  it('merges concurrent setValues updates without losing fields', () => {
    const { result } = renderHook(() =>
      useFormValidation({
        first: { initialValue: '' },
        second: { initialValue: '' },
        third: { initialValue: '' },
      })
    );

    act(() => {
      result.current.setValues({ first: 'one', second: 'two' });
      result.current.setValues({ third: 'three' });
    });

    expect(result.current.values).toEqual({
      first: 'one',
      second: 'two',
      third: 'three',
    });
  });
});
