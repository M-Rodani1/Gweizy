/**
 * Tests for ModelTrainingPanel component
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import ModelTrainingPanel from '../../components/ModelTrainingPanel';

// Mock fetch globally
global.fetch = vi.fn();

// Mock confirm dialog
global.confirm = vi.fn();

describe('ModelTrainingPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    (global.confirm as any).mockReturnValue(true);
  });

  afterEach(() => {
    vi.resetAllMocks();
    vi.useRealTimers();
  });

  const mockModelsStatusResponse = {
    prediction_models: {
      '1h': { available: true, path: '/models/prediction_1h.pkl' },
      '4h': { available: true, path: '/models/prediction_4h.pkl' },
      '24h': { available: false, path: null }
    },
    spike_detectors: {
      '1h': { available: true, path: '/models/spike_1h.pkl' },
      '4h': { available: false, path: null },
      '24h': { available: false, path: null }
    },
    dqn_agent: { available: false, path: null },
    data_status: {
      total_records: 1500,
      sufficient_for_training: true,
      sufficient_for_dqn: true
    },
    overall_ready: false,
    missing_models: ['prediction_24h', 'spike_4h', 'spike_24h', 'dqn_agent'],
    summary: {
      prediction_models_ready: false,
      spike_detectors_ready: false,
      dqn_agent_ready: false,
      action_needed: 'Train missing models'
    }
  };

  const mockAllReadyResponse = {
    ...mockModelsStatusResponse,
    prediction_models: {
      '1h': { available: true, path: '/models/prediction_1h.pkl' },
      '4h': { available: true, path: '/models/prediction_4h.pkl' },
      '24h': { available: true, path: '/models/prediction_24h.pkl' }
    },
    spike_detectors: {
      '1h': { available: true, path: '/models/spike_1h.pkl' },
      '4h': { available: true, path: '/models/spike_4h.pkl' },
      '24h': { available: true, path: '/models/spike_24h.pkl' }
    },
    dqn_agent: { available: true, path: '/models/dqn_agent.pkl' },
    overall_ready: true,
    missing_models: [],
    summary: {
      prediction_models_ready: true,
      spike_detectors_ready: true,
      dqn_agent_ready: true,
      action_needed: null
    }
  };

  const mockTrainingProgress = {
    is_training: false,
    current_step: 0,
    total_steps: 3,
    step_name: null,
    step_status: null,
    steps: [
      { name: 'RandomForest Models', status: 'pending', message: null },
      { name: 'Spike Detectors', status: 'pending', message: null },
      { name: 'DQN Agent', status: 'pending', message: null }
    ],
    started_at: null,
    completed_at: null,
    error: null
  };

  const mockActiveTrainingProgress = {
    is_training: true,
    current_step: 1,
    total_steps: 3,
    step_name: 'Spike Detectors',
    step_status: 'running',
    steps: [
      { name: 'RandomForest Models', status: 'completed', message: 'Trained 3 models' },
      { name: 'Spike Detectors', status: 'running', message: 'Training in progress...' },
      { name: 'DQN Agent', status: 'pending', message: null }
    ],
    started_at: new Date().toISOString(),
    completed_at: null,
    error: null
  };

  it('renders loading state initially', () => {
    (global.fetch as any).mockImplementation(() => new Promise(() => {}));

    render(<ModelTrainingPanel />);

    // Should show loading spinner (animate-spin class)
    expect(document.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders header after successful fetch', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('ML Model Training')).toBeInTheDocument();
    });
  });

  it('displays model status sections', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Prediction Models')).toBeInTheDocument();
      expect(screen.getByText('Spike Detectors')).toBeInTheDocument();
      expect(screen.getByText('DQN Agent')).toBeInTheDocument();
      expect(screen.getByText('Training Data')).toBeInTheDocument();
    });
  });

  it('shows "Models Need Training" when not all models ready', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Models Need Training')).toBeInTheDocument();
    });
  });

  it('shows "All Models Ready" when all models available', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockAllReadyResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('All Models Ready')).toBeInTheDocument();
    });
  });

  it('displays training data record count', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('1,500 records')).toBeInTheDocument();
    });
  });

  it('shows "Train All Models" button when models missing', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Train All Models/i })).toBeInTheDocument();
    });
  });

  it('shows "Retrain All Models" button when all models ready', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockAllReadyResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Retrain All Models/i })).toBeInTheDocument();
    });
  });

  it('renders refresh button', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });
  });

  it('calls fetch on refresh button click', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });

    // Clear previous calls
    vi.clearAllMocks();

    // Click refresh
    fireEvent.click(screen.getByText('Refresh'));

    // Fetch should be called again
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });
  });

  it('shows error state when fetch fails', async () => {
    // Mock all fetch calls to fail
    (global.fetch as any).mockImplementation(() => {
      return Promise.reject(new Error('Network error'));
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      // The component displays the actual error message
      expect(screen.getByText('Network error')).toBeInTheDocument();
    }, { timeout: 3000 });
  });

  it('shows retry button in error state', async () => {
    // Mock all fetch calls to fail
    (global.fetch as any).mockImplementation(() => {
      return Promise.reject(new Error('Network error'));
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
    }, { timeout: 3000 });
  });

  it('triggers training when button clicked and confirmed', async () => {
    (global.fetch as any).mockImplementation((url: string, options?: any) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      if (url.includes('retraining/simple') && options?.method === 'POST') {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'started', message: 'Training started' })
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Train All Models/i })).toBeInTheDocument();
    });

    // Click train button
    fireEvent.click(screen.getByRole('button', { name: /Train All Models/i }));

    // Confirm should have been called
    expect(global.confirm).toHaveBeenCalled();

    // Check that training API was called
    await waitFor(() => {
      const calls = (global.fetch as any).mock.calls;
      const trainingCall = calls.find((call: any[]) =>
        call[0].includes('retraining/simple') && call[1]?.method === 'POST'
      );
      expect(trainingCall).toBeTruthy();
    });
  });

  it('does not trigger training when confirm is cancelled', async () => {
    (global.confirm as any).mockReturnValue(false);

    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Train All Models/i })).toBeInTheDocument();
    });

    vi.clearAllMocks();

    // Click train button
    fireEvent.click(screen.getByRole('button', { name: /Train All Models/i }));

    // Training API should NOT be called
    const calls = (global.fetch as any).mock.calls;
    const trainingCall = calls.find((call: any[]) =>
      call[0]?.includes('retraining/simple')
    );
    expect(trainingCall).toBeFalsy();
  });

  it('displays training progress when training is active', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockActiveTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Training in Progress')).toBeInTheDocument();
    });
  });

  it('displays step names in training progress', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockActiveTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    // First wait for the training in progress header to appear
    await waitFor(() => {
      expect(screen.getByText('Training in Progress')).toBeInTheDocument();
    }, { timeout: 3000 });

    // Then check for step names (RandomForest Models is unique to training progress section)
    expect(screen.getByText('RandomForest Models')).toBeInTheDocument();
  });

  it('shows step status indicators', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockActiveTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      // Check for status text
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('pending')).toBeInTheDocument();
    });
  });

  it('shows step message when available', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockActiveTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Trained 3 models')).toBeInTheDocument();
      expect(screen.getByText('Training in progress...')).toBeInTheDocument();
    });
  });

  it('displays horizon badges for prediction models', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      // Check for horizon badges (appears twice - once for prediction, once for spike)
      const badges1h = screen.getAllByText('1h');
      const badges4h = screen.getAllByText('4h');
      const badges24h = screen.getAllByText('24h');

      expect(badges1h.length).toBeGreaterThanOrEqual(2);
      expect(badges4h.length).toBeGreaterThanOrEqual(2);
      expect(badges24h.length).toBeGreaterThanOrEqual(2);
    });
  });

  it('disables train button when insufficient data', async () => {
    const insufficientDataResponse = {
      ...mockModelsStatusResponse,
      data_status: {
        total_records: 30,
        sufficient_for_training: false,
        sufficient_for_dqn: false
      }
    };

    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => insufficientDataResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      const trainButton = screen.getByRole('button', { name: /Train All Models/i });
      expect(trainButton).toBeDisabled();
    });
  });

  it('shows warning message when insufficient data', async () => {
    const insufficientDataResponse = {
      ...mockModelsStatusResponse,
      data_status: {
        total_records: 30,
        sufficient_for_training: false,
        sufficient_for_dqn: false
      }
    };

    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => insufficientDataResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText(/Need at least 50 data records/i)).toBeInTheDocument();
    });
  });

  it('shows data status error when database check fails', async () => {
    const errorDataResponse = {
      ...mockModelsStatusResponse,
      data_status: {
        total_records: 0,
        sufficient_for_training: false,
        sufficient_for_dqn: false,
        error: 'Database connection failed'
      }
    };

    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => errorDataResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Database connection failed')).toBeInTheDocument();
    });
  });

  it('displays last updated time', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText(/Last check:/i)).toBeInTheDocument();
    });
  });

  it('shows missing model count', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('4 missing')).toBeInTheDocument();
    });
  });

  it('shows training error when training fails', async () => {
    // Error during active training (is_training must be true for progress section to show)
    const errorProgress = {
      ...mockActiveTrainingProgress,
      is_training: true,
      error: 'Training failed: Out of memory',
      steps: [
        { name: 'RandomForest Models', status: 'completed', message: 'Trained successfully' },
        { name: 'Spike Detectors', status: 'failed', message: 'Out of memory' },
        { name: 'DQN Agent', status: 'pending', message: null }
      ]
    };

    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => errorProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Training failed: Out of memory')).toBeInTheDocument();
    }, { timeout: 3000 });
  });

  it('shows step counter during training', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('models-status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockModelsStatusResponse
        });
      }
      if (url.includes('training-progress')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockActiveTrainingProgress
        });
      }
      return Promise.resolve({ ok: false });
    });

    render(<ModelTrainingPanel />);

    await waitFor(() => {
      expect(screen.getByText('Step 2/3')).toBeInTheDocument();
    });
  });
});
