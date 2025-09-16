import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { ConfigWizard } from '../ConfigWizard';
import '@testing-library/jest-dom';

// Test utilities
const renderWithRouter = (component: React.ReactElement) => {
  return render(<BrowserRouter>{component}</BrowserRouter>);
};

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn()
};

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
  writable: true
});

// Mock fetch for API calls
global.fetch = jest.fn();

describe('ConfigWizard Integration Tests', () => {
  const mockOnComplete = jest.fn();
  const mockOnCancel = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
    (global.fetch as jest.Mock).mockImplementation(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ success: true })
      })
    );
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Complete Wizard Flow', () => {
    it('should complete entire configuration wizard successfully', async () => {
      const user = userEvent.setup();

      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Step 1: Environment Configuration
      expect(screen.getByText('Environment')).toBeInTheDocument();

      // Fill environment form
      const appNameInput = screen.getByLabelText(/Application Name/i);
      await user.type(appNameInput, 'test-brain-app');

      const baseUrlInput = screen.getByLabelText(/Base URL/i);
      await user.clear(baseUrlInput);
      await user.type(baseUrlInput, 'http://localhost:3000');

      // Go to next step
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('Database')).toBeInTheDocument();
      });

      // Step 2: Database Configuration
      const dbNameInput = screen.getByLabelText(/Database Name/i);
      await user.type(dbNameInput, 'brain_test_db');

      const dbUserInput = screen.getByLabelText(/Username/i);
      await user.type(dbUserInput, 'postgres');

      const dbPasswordInput = screen.getByLabelText(/Password/i);
      await user.type(dbPasswordInput, 'password123');

      // Go to next step
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('API Keys')).toBeInTheDocument();
      });

      // Step 3: API Keys Configuration
      const openaiKeyInput = screen.getByLabelText(/OpenAI API Key/i);
      await user.type(openaiKeyInput, 'sk-test-openai-key');

      const supabaseUrlInput = screen.getByLabelText(/Supabase URL/i);
      await user.type(supabaseUrlInput, 'https://test.supabase.co');

      const supabaseKeyInput = screen.getByLabelText(/Supabase Anon Key/i);
      await user.type(supabaseKeyInput, 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...');

      // Go to next step
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('Services')).toBeInTheDocument();
      });

      // Step 4: Services Configuration - toggle some services
      const serviceToggles = screen.getAllByRole('button');
      const realtimeToggle = serviceToggles.find(btn =>
        btn.closest('div')?.textContent?.includes('Real-time Updates')
      );
      if (realtimeToggle) {
        fireEvent.click(realtimeToggle);
      }

      // Go to final step
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('Review & Test')).toBeInTheDocument();
      });

      // Step 5: Review and Complete
      expect(screen.getByText('Configuration Summary')).toBeInTheDocument();
      expect(screen.getByText('test-brain-app')).toBeInTheDocument();
      expect(screen.getByText('brain_test_db')).toBeInTheDocument();

      // Complete the wizard
      fireEvent.click(screen.getByText('Complete Configuration'));

      await waitFor(() => {
        expect(mockOnComplete).toHaveBeenCalledWith(
          expect.objectContaining({
            appName: 'test-brain-app',
            baseUrl: 'http://localhost:3000',
            database: expect.objectContaining({
              name: 'brain_test_db',
              username: 'postgres'
            }),
            apiKeys: expect.objectContaining({
              openai: 'sk-test-openai-key',
              supabase_url: 'https://test.supabase.co'
            })
          })
        );
      });
    });

    it('should handle validation errors and prevent progression', async () => {
      const user = userEvent.setup();

      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Try to proceed without filling required fields
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('Please fix the following errors:')).toBeInTheDocument();
        expect(screen.getByText(/Application name is required/)).toBeInTheDocument();
      });

      // Should still be on first step
      expect(screen.getByText('Environment')).toBeInTheDocument();
    });

    it('should save and restore progress from localStorage', async () => {
      const user = userEvent.setup();
      const savedProgress = {
        currentStep: 2,
        config: {
          appName: 'saved-app',
          baseUrl: 'http://saved.com',
          database: { name: 'saved_db' }
        },
        completedSteps: [0, 1],
        timestamp: Date.now()
      };

      localStorageMock.getItem.mockReturnValue(JSON.stringify(savedProgress));

      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Should load saved progress
      expect(localStorageMock.getItem).toHaveBeenCalledWith('brain_config_wizard_progress');

      // Fill some data and expect auto-save
      const appNameInput = screen.getByLabelText(/Application Name/i);
      await user.type(appNameInput, 'new-name');

      await waitFor(() => {
        expect(localStorageMock.setItem).toHaveBeenCalledWith(
          'brain_config_wizard_progress',
          expect.stringContaining('new-name')
        );
      }, { timeout: 2000 });
    });
  });

  describe('Navigation Flows', () => {
    it('should allow backward navigation while preserving data', async () => {
      const user = userEvent.setup();

      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Fill first step
      await user.type(screen.getByLabelText(/Application Name/i), 'test-app');
      await user.clear(screen.getByLabelText(/Base URL/i));
      await user.type(screen.getByLabelText(/Base URL/i), 'http://test.com');

      // Go to second step
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('Database')).toBeInTheDocument();
      });

      // Fill second step
      await user.type(screen.getByLabelText(/Database Name/i), 'test_db');

      // Go back to first step
      fireEvent.click(screen.getByText('Previous'));

      await waitFor(() => {
        expect(screen.getByText('Environment')).toBeInTheDocument();
      });

      // Data should be preserved
      expect(screen.getByDisplayValue('test-app')).toBeInTheDocument();
      expect(screen.getByDisplayValue('http://test.com')).toBeInTheDocument();

      // Go forward again
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('Database')).toBeInTheDocument();
      });

      // Database data should also be preserved
      expect(screen.getByDisplayValue('test_db')).toBeInTheDocument();
    });

    it('should handle cancellation at any step', async () => {
      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Cancel from first step
      fireEvent.click(screen.getByText('Cancel'));
      expect(mockOnCancel).toHaveBeenCalled();

      // Reset mock and test from later step
      mockOnCancel.mockClear();

      // Navigate to a later step
      fireEvent.click(screen.getByText('Next'));
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('API Keys')).toBeInTheDocument();
      });

      // Cancel from API Keys step
      fireEvent.click(screen.getByText('Cancel'));
      expect(mockOnCancel).toHaveBeenCalled();
    });
  });

  describe('AI Assistant Integration', () => {
    it('should toggle AI assistant panel', async () => {
      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Initially hidden
      expect(screen.queryByText('AI Configuration Assistant')).not.toBeInTheDocument();

      // Show assistant
      fireEvent.click(screen.getByText(/Show AI Assistant/));
      expect(screen.getByText('AI Configuration Assistant')).toBeInTheDocument();

      // Hide assistant
      fireEvent.click(screen.getByText(/Hide AI Assistant/));
      expect(screen.queryByText('AI Configuration Assistant')).not.toBeInTheDocument();
    });

    it('should provide contextual help based on current step', async () => {
      const user = userEvent.setup();

      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Show AI assistant
      fireEvent.click(screen.getByText(/Show AI Assistant/));

      // Go to database step
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('Database')).toBeInTheDocument();
      });

      // Ask AI about database
      const chatInput = screen.getByPlaceholderText('Ask me anything about configuration...');
      await user.type(chatInput, 'How do I configure the database?{enter}');

      await waitFor(() => {
        expect(screen.getByText(/PostgreSQL/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });

  describe('Error Recovery', () => {
    it('should recover from API errors gracefully', async () => {
      // Mock API failure
      (global.fetch as jest.Mock).mockImplementationOnce(() =>
        Promise.resolve({
          ok: false,
          json: () => Promise.resolve({ error: 'Database connection failed' })
        })
      );

      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Navigate to database step
      fireEvent.click(screen.getByText('Next'));

      await waitFor(() => {
        expect(screen.getByText('Database')).toBeInTheDocument();
      });

      // Trigger a database test that fails
      const testButton = screen.getByText('Test Connection');
      fireEvent.click(testButton);

      await waitFor(() => {
        expect(screen.getByText(/Connection failed/)).toBeInTheDocument();
      });

      // Should still allow navigation
      expect(screen.getByText('Next')).toBeInTheDocument();
    });

    it('should handle localStorage errors gracefully', async () => {
      // Mock localStorage failure
      localStorageMock.setItem.mockImplementation(() => {
        throw new Error('Storage quota exceeded');
      });

      const user = userEvent.setup();

      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Should not crash when trying to auto-save
      await user.type(screen.getByLabelText(/Application Name/i), 'test');

      // Wait for auto-save attempt
      await waitFor(() => {
        expect(localStorageMock.setItem).toHaveBeenCalled();
      }, { timeout: 2000 });

      // Wizard should still be functional
      expect(screen.getByText('Environment')).toBeInTheDocument();
    });
  });

  describe('Configuration Export', () => {
    it('should generate correct environment variables in review step', async () => {
      const user = userEvent.setup();

      renderWithRouter(
        <ConfigWizard
          onComplete={mockOnComplete}
          onCancel={mockOnCancel}
        />
      );

      // Fill minimal configuration
      await user.type(screen.getByLabelText(/Application Name/i), 'export-test');
      await user.clear(screen.getByLabelText(/Base URL/i));
      await user.type(screen.getByLabelText(/Base URL/i), 'http://export.test');

      // Navigate to review step
      for (let i = 0; i < 4; i++) {
        fireEvent.click(screen.getByText(i === 3 ? 'Next' : 'Next'));
        await waitFor(() => {});
      }

      // Check configuration summary
      expect(screen.getByText('export-test')).toBeInTheDocument();
      expect(screen.getByText('http://export.test')).toBeInTheDocument();

      // Show environment variables
      fireEvent.click(screen.getByText('📄'));

      await waitFor(() => {
        expect(screen.getByText(/APP_NAME=export-test/)).toBeInTheDocument();
        expect(screen.getByText(/BASE_URL=http:\/\/export\.test/)).toBeInTheDocument();
      });
    });
  });
});