import { render, screen } from '@testing-library/react';
import { ToastProvider } from '../../providers/ToastProvider';

vi.mock('react-hot-toast', () => ({
  Toaster: () => <div data-testid="toaster" />,
  useToasterStore: () => ({
    toasts: [
      { id: '1', visible: true, message: 'Gas price updated' },
      { id: '2', visible: false, message: 'Hidden' },
    ],
  }),
}));

describe('ToastProvider live region', () => {
  it('announces visible toast messages', () => {
    render(
      <ToastProvider>
        <div>Child</div>
      </ToastProvider>
    );

    const liveRegion = screen.getByRole('status');
    expect(liveRegion).toHaveTextContent('Gas price updated');
    expect(liveRegion).not.toHaveTextContent('Hidden');
  });
});
