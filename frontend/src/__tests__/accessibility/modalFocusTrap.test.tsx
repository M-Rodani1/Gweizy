import { fireEvent, render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import { Modal, ModalBody, ModalHeader } from '../../components/ui/Modal';

describe('Modal focus management', () => {
  it('traps focus within the modal', () => {
    const rafSpy = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => {
        cb(0);
        return 0;
      });

    render(
      <Modal isOpen onClose={vi.fn()}>
        <ModalHeader>Focus</ModalHeader>
        <ModalBody>
          <button type="button">First action</button>
          <button type="button">Last action</button>
        </ModalBody>
      </Modal>
    );

    const lastButton = screen.getByRole('button', { name: 'Last action' });
    lastButton.focus();
    fireEvent.keyDown(document, { key: 'Tab' });

    const closeButton = screen.getByRole('button', { name: 'Close modal' });
    expect(document.activeElement).toBe(closeButton);

    rafSpy.mockRestore();
  });

  it('restores focus when the modal closes', () => {
    const rafSpy = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => {
        cb(0);
        return 0;
      });

    const { rerender } = render(
      <>
        <button type="button">Open modal</button>
        <Modal isOpen={false} onClose={vi.fn()}>
          <ModalHeader>Focus</ModalHeader>
          <ModalBody>
            <button type="button">First action</button>
          </ModalBody>
        </Modal>
      </>
    );

    const trigger = screen.getByRole('button', { name: 'Open modal' });
    trigger.focus();

    rerender(
      <>
        <button type="button">Open modal</button>
        <Modal isOpen onClose={vi.fn()}>
          <ModalHeader>Focus</ModalHeader>
          <ModalBody>
            <button type="button">First action</button>
          </ModalBody>
        </Modal>
      </>
    );

    rerender(
      <>
        <button type="button">Open modal</button>
        <Modal isOpen={false} onClose={vi.fn()}>
          <ModalHeader>Focus</ModalHeader>
          <ModalBody>
            <button type="button">First action</button>
          </ModalBody>
        </Modal>
      </>
    );

    expect(document.activeElement).toBe(trigger);

    rafSpy.mockRestore();
  });
});
