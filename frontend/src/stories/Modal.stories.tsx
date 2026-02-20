/**
 * Storybook stories for Modal component
 */

import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';
import { Modal, ModalHeader, ModalBody, ModalFooter } from '../components/ui/Modal';
import { Button } from '../components/ui/Button';

const meta: Meta<typeof Modal> = {
  title: 'Components/Modal',
  component: Modal,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg', 'xl', 'full'],
    },
    closeOnOverlayClick: {
      control: 'boolean',
    },
    closeOnEscape: {
      control: 'boolean',
    },
    showCloseButton: {
      control: 'boolean',
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

const ModalDemo = ({ size = 'lg', ...props }: { size?: 'sm' | 'md' | 'lg' | 'xl' | 'full' }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <Button onClick={() => setIsOpen(true)}>Open Modal</Button>
      <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} size={size} {...props}>
        <ModalHeader subtitle="This is a subtitle">Modal Title</ModalHeader>
        <ModalBody>
          <p className="text-gray-300">
            This is the modal body content. You can put any content here including forms,
            tables, or other components.
          </p>
          <div className="bg-gray-800 rounded-lg p-4 mt-4">
            <p className="text-gray-400 text-sm">
              Press Escape to close, or click outside the modal.
            </p>
          </div>
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" onClick={() => setIsOpen(false)}>Cancel</Button>
          <Button variant="primary" onClick={() => setIsOpen(false)}>Confirm</Button>
        </ModalFooter>
      </Modal>
    </>
  );
};

const WithFormDemo = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <Button onClick={() => setIsOpen(true)}>Open Form Modal</Button>
      <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} size="md">
        <ModalHeader>Set Gas Alert</ModalHeader>
        <ModalBody>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Target Gas Price</label>
              <input
                type="number"
                placeholder="Enter gwei value"
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-2">Notification Method</label>
              <select className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white">
                <option>Browser notification</option>
                <option>Email</option>
                <option>Both</option>
              </select>
            </div>
          </div>
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" onClick={() => setIsOpen(false)}>Cancel</Button>
          <Button variant="primary" onClick={() => setIsOpen(false)}>Set Alert</Button>
        </ModalFooter>
      </Modal>
    </>
  );
};

const NoCloseButtonDemo = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <Button onClick={() => setIsOpen(true)}>Open Modal Without Close</Button>
      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        showCloseButton={false}
        closeOnOverlayClick={false}
      >
        <ModalHeader>Important Action</ModalHeader>
        <ModalBody>
          <p className="text-gray-300">
            This modal requires you to make a choice before closing.
          </p>
        </ModalBody>
        <ModalFooter>
          <Button variant="danger" onClick={() => setIsOpen(false)}>Reject</Button>
          <Button variant="success" onClick={() => setIsOpen(false)}>Accept</Button>
        </ModalFooter>
      </Modal>
    </>
  );
};

export const Default: Story = {
  render: () => <ModalDemo />,
};

export const SmallSize: Story = {
  render: () => <ModalDemo size="sm" />,
};

export const FullSize: Story = {
  render: () => <ModalDemo size="full" />,
};

export const WithForm: Story = {
  render: () => <WithFormDemo />,
};

export const NoCloseButton: Story = {
  render: () => <NoCloseButtonDemo />,
};
