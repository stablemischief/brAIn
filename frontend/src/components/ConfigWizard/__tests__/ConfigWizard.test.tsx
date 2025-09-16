import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConfigWizard } from '../ConfigWizard';
import '@testing-library/jest-dom';

// Mock the child components
jest.mock('../../ConfigSteps/EnvironmentStep', () => ({
  EnvironmentStep: ({ config, onUpdate }: any) => (
    <div data-testid="environment-step">
      <input
        data-testid="app-name-input"
        value={config.appName || ''}
        onChange={(e) => onUpdate({ ...config, appName: e.target.value })}
      />
    </div>
  )
}));

jest.mock('../../ConfigSteps/DatabaseStep', () => ({
  DatabaseStep: () => <div data-testid="database-step">Database Step</div>
}));

jest.mock('../../ConfigSteps/APIKeysStep', () => ({
  APIKeysStep: () => <div data-testid="api-keys-step">API Keys Step</div>
}));

jest.mock('../../ConfigSteps/ServicesStep', () => ({
  ServicesStep: () => <div data-testid="services-step">Services Step</div>
}));

jest.mock('../../ConfigSteps/ReviewStep', () => ({
  ReviewStep: ({ onTest }: any) => (
    <div data-testid="review-step">
      <button onClick={onTest}>Complete Setup</button>
    </div>
  )
}));

// Create a mock hook that doesn't reference external variables
const mockSetCurrentStep = jest.fn();
const mockSetConfig = jest.fn();
const mockValidateStep = jest.fn().mockResolvedValue(true);
const mockSaveProgress = jest.fn();
const mockLoadProgress = jest.fn();
const mockClearProgress = jest.fn();

jest.mock('../../../hooks/useConfigWizard', () => ({
  useConfigWizard: jest.fn()
}));

describe('ConfigWizard', () => {
  const mockOnComplete = jest.fn();
  const mockOnCancel = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();

    // Setup the mock hook implementation
    const { useConfigWizard } = require('../../../hooks/useConfigWizard');
    useConfigWizard.mockImplementation((initialConfig: any) => ({
      currentStep: 0,
      config: initialConfig || {},
      errors: {},
      isValidating: false,
      setCurrentStep: mockSetCurrentStep,
      updateConfig: mockSetConfig,
      validateStep: mockValidateStep,
      saveProgress: mockSaveProgress,
      loadProgress: mockLoadProgress,
      clearProgress: mockClearProgress
    }));
  });

  describe('Component Rendering', () => {
    it('should render the wizard with initial step', () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Configuration Wizard')).toBeInTheDocument();
      expect(screen.getByText("Let's set up your brAIn system step by step")).toBeInTheDocument();
      expect(screen.getByTestId('environment-step')).toBeInTheDocument();
    });

    it('should display step progress indicator', () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Step 1 of 5')).toBeInTheDocument();
    });

    it('should show current step icon and label', () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Environment')).toBeInTheDocument();
      expect(screen.getByText('🌍')).toBeInTheDocument();
    });
  });

  describe('Navigation', () => {
    it('should navigate to next step when Next is clicked', async () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      const nextButton = screen.getByText('Next');
      fireEvent.click(nextButton);

      await waitFor(() => {
        expect(screen.getByTestId('database-step')).toBeInTheDocument();
      });
    });

    it('should navigate to previous step when Previous is clicked', async () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Go to step 2
      fireEvent.click(screen.getByText('Next'));
      await waitFor(() => {
        expect(screen.getByTestId('database-step')).toBeInTheDocument();
      });

      // Go back to step 1
      fireEvent.click(screen.getByText('Previous'));
      await waitFor(() => {
        expect(screen.getByTestId('environment-step')).toBeInTheDocument();
      });
    });

    it('should not show Previous button on first step', () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.queryByText('Previous')).not.toBeInTheDocument();
    });

    it('should show Complete Setup button on last step', async () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Navigate to last step
      for (let i = 0; i < 4; i++) {
        fireEvent.click(screen.getByText(i === 3 ? 'Next' : 'Next'));
        await waitFor(() => {});
      }

      expect(screen.getByText('Complete Setup')).toBeInTheDocument();
    });
  });

  describe('Configuration Updates', () => {
    it('should update configuration when step input changes', async () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      const appNameInput = screen.getByTestId('app-name-input');
      await userEvent.type(appNameInput, 'test-app');

      expect(appNameInput).toHaveValue('test-app');
    });

    it('should preserve configuration when navigating between steps', async () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Set a value in step 1
      const appNameInput = screen.getByTestId('app-name-input');
      await userEvent.type(appNameInput, 'test-app');

      // Go to step 2 and back
      fireEvent.click(screen.getByText('Next'));
      await waitFor(() => {
        expect(screen.getByTestId('database-step')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('Previous'));
      await waitFor(() => {
        expect(screen.getByTestId('environment-step')).toBeInTheDocument();
      });

      // Value should be preserved
      expect(screen.getByTestId('app-name-input')).toHaveValue('test-app');
    });
  });

  describe('Completion and Cancellation', () => {
    it('should call onComplete when wizard is completed', async () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Navigate to last step
      for (let i = 0; i < 4; i++) {
        fireEvent.click(screen.getByText('Next'));
        await waitFor(() => {});
      }

      // Complete the wizard
      fireEvent.click(screen.getByText('Complete Setup'));

      await waitFor(() => {
        expect(mockOnComplete).toHaveBeenCalled();
      });
    });

    it('should call onCancel when Cancel is clicked', () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      fireEvent.click(screen.getByText('Cancel'));
      expect(mockOnCancel).toHaveBeenCalled();
    });
  });

  describe('AI Assistant Integration', () => {
    it('should toggle AI assistant visibility', () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      const toggleButton = screen.getByText(/AI Assistant/);

      // Initially hidden
      expect(screen.queryByText('AI Assistant coming soon...')).not.toBeInTheDocument();

      // Show assistant
      fireEvent.click(toggleButton);
      expect(screen.getByText('AI Configuration Assistant')).toBeInTheDocument();

      // Hide assistant
      fireEvent.click(screen.getByText('✕'));
      expect(screen.queryByText('AI Configuration Assistant')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should display error summary when validation fails', async () => {
      const { useConfigWizard } = require('../../../hooks/useConfigWizard');
      useConfigWizard.mockImplementation(() => ({
        currentStep: 0,
        config: {},
        errors: { appName: 'Application name is required' },
        isValidating: false,
        setCurrentStep: jest.fn(),
        updateConfig: jest.fn(),
        validateStep: jest.fn().mockResolvedValue(false),
        saveProgress: jest.fn(),
        loadProgress: jest.fn(),
        clearProgress: jest.fn()
      }));

      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Please fix the following errors:')).toBeInTheDocument();
      expect(screen.getByText('• Application name is required')).toBeInTheDocument();
    });
  });

  describe('Progress Persistence', () => {
    it('should auto-save progress periodically', async () => {
      const mockSaveProgress = jest.fn();
      const { useConfigWizard } = require('../../../hooks/useConfigWizard');

      useConfigWizard.mockImplementation(() => ({
        currentStep: 0,
        config: { appName: 'test' },
        errors: {},
        isValidating: false,
        setCurrentStep: jest.fn(),
        updateConfig: jest.fn(),
        validateStep: jest.fn().mockResolvedValue(true),
        saveProgress: mockSaveProgress,
        loadProgress: jest.fn(),
        clearProgress: jest.fn()
      }));

      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Wait for auto-save timeout
      await waitFor(() => {
        expect(mockSaveProgress).toHaveBeenCalled();
      }, { timeout: 2000 });
    });
  });

  describe('Template Tip', () => {
    it('should show template tip on first step', () => {
      render(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText(/You can use a template/)).toBeInTheDocument();
      expect(screen.getByText('Browse Templates')).toBeInTheDocument();
    });
  });
});