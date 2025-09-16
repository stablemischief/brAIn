import { renderHook, act } from '@testing-library/react';
import { useConfigWizard } from '../useConfigWizard';

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

describe('useConfigWizard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
  });

  describe('Initial State', () => {
    it('should initialize with default configuration', () => {
      const { result } = renderHook(() => useConfigWizard());

      expect(result.current.currentStep).toBe(0);
      expect(result.current.config.environment).toBe('development');
      expect(result.current.config.backendPort).toBe(8000);
      expect(result.current.config.frontendPort).toBe(3000);
      expect(result.current.errors).toEqual({});
      expect(result.current.isValidating).toBe(false);
    });

    it('should accept initial configuration', () => {
      const initialConfig = {
        environment: 'production',
        appName: 'test-app'
      };

      const { result } = renderHook(() => useConfigWizard(initialConfig));

      expect(result.current.config.environment).toBe('production');
      expect(result.current.config.appName).toBe('test-app');
    });
  });

  describe('Step Navigation', () => {
    it('should update current step', () => {
      const { result } = renderHook(() => useConfigWizard());

      act(() => {
        result.current.setCurrentStep(2);
      });

      expect(result.current.currentStep).toBe(2);
    });
  });

  describe('Configuration Updates', () => {
    it('should update configuration', () => {
      const { result } = renderHook(() => useConfigWizard());

      act(() => {
        result.current.updateConfig({ appName: 'new-app', environment: 'staging' });
      });

      expect(result.current.config.appName).toBe('new-app');
      expect(result.current.config.environment).toBe('staging');
    });

    it('should merge configuration updates', () => {
      const { result } = renderHook(() => useConfigWizard());

      act(() => {
        result.current.updateConfig({ appName: 'app1' });
      });

      act(() => {
        result.current.updateConfig({ environment: 'production' });
      });

      expect(result.current.config.appName).toBe('app1');
      expect(result.current.config.environment).toBe('production');
    });
  });

  describe('Step Validation', () => {
    describe('Environment Step (0)', () => {
      it('should validate required app name', async () => {
        const { result } = renderHook(() => useConfigWizard());

        act(() => {
          result.current.updateConfig({ appName: '' });
        });

        let validationResult: boolean = false;
        await act(async () => {
          validationResult = await result.current.validateStep(0);
        });

        expect(validationResult).toBe(false);
        expect(result.current.errors.appName).toBe('Application name is required');
      });

      it('should validate required base URL', async () => {
        const { result } = renderHook(() => useConfigWizard());

        act(() => {
          result.current.updateConfig({ appName: 'test', baseUrl: '' });
        });

        let validationResult: boolean = false;
        await act(async () => {
          validationResult = await result.current.validateStep(0);
        });

        expect(validationResult).toBe(false);
        expect(result.current.errors.baseUrl).toBe('Base URL is required');
      });

      it('should validate port conflicts', async () => {
        const { result } = renderHook(() => useConfigWizard());

        act(() => {
          result.current.updateConfig({
            appName: 'test',
            baseUrl: 'http://localhost',
            backendPort: 3000,
            frontendPort: 3000
          });
        });

        let validationResult: boolean = false;
        await act(async () => {
          validationResult = await result.current.validateStep(0);
        });

        expect(validationResult).toBe(false);
        expect(result.current.errors.backendPort).toBe('Backend and frontend ports must be different');
      });
    });

    describe('Database Step (1)', () => {
      it('should validate database configuration for non-SQLite', async () => {
        const { result } = renderHook(() => useConfigWizard());

        act(() => {
          result.current.updateConfig({
            database: { type: 'postgresql' }
          });
        });

        let validationResult: boolean = false;
        await act(async () => {
          validationResult = await result.current.validateStep(1);
        });

        expect(validationResult).toBe(false);
        expect(result.current.errors['database.host']).toBe('Database host is required');
        expect(result.current.errors['database.name']).toBe('Database name is required');
        expect(result.current.errors['database.username']).toBe('Database username is required');
      });

      it('should validate SQLite path', async () => {
        const { result } = renderHook(() => useConfigWizard());

        act(() => {
          result.current.updateConfig({
            database: { type: 'sqlite', path: '' }
          });
        });

        let validationResult: boolean = false;
        await act(async () => {
          validationResult = await result.current.validateStep(1);
        });

        expect(validationResult).toBe(false);
        expect(result.current.errors['database.path']).toBe('Database file path is required');
      });
    });

    describe('API Keys Step (2)', () => {
      it('should validate required API keys', async () => {
        const { result } = renderHook(() => useConfigWizard());

        act(() => {
          result.current.updateConfig({ apiKeys: {} });
        });

        let validationResult: boolean = false;
        await act(async () => {
          validationResult = await result.current.validateStep(2);
        });

        expect(validationResult).toBe(false);
        expect(result.current.errors['apiKeys.openai']).toBe('OpenAI API key is required');
        expect(result.current.errors['apiKeys.supabase_url']).toBe('Supabase URL is required');
        expect(result.current.errors['apiKeys.supabase_key']).toBe('Supabase key is required');
      });
    });

    it('should add step to completed steps on successful validation', async () => {
      const { result } = renderHook(() => useConfigWizard());

      act(() => {
        result.current.updateConfig({
          appName: 'test',
          baseUrl: 'http://localhost'
        });
      });

      await act(async () => {
        await result.current.validateStep(0);
      });

      expect(result.current.completedSteps).toContain(0);
    });
  });

  describe('Progress Persistence', () => {
    it('should save progress to localStorage', () => {
      const { result } = renderHook(() => useConfigWizard());

      act(() => {
        result.current.updateConfig({ appName: 'test-app' });
        result.current.setCurrentStep(2);
      });

      act(() => {
        result.current.saveProgress();
      });

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'brain_config_wizard_progress',
        expect.stringContaining('"appName":"test-app"')
      );
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'brain_config_wizard_progress',
        expect.stringContaining('"currentStep":2')
      );
    });

    it('should load progress from localStorage', () => {
      const savedData = {
        currentStep: 3,
        config: { appName: 'saved-app' },
        completedSteps: [0, 1],
        timestamp: Date.now()
      };

      localStorageMock.getItem.mockReturnValue(JSON.stringify(savedData));

      const { result } = renderHook(() => useConfigWizard());

      const loaded = result.current.loadProgress();

      expect(loaded).toEqual(savedData);
    });

    it('should clear progress from localStorage', () => {
      const { result } = renderHook(() => useConfigWizard());

      act(() => {
        result.current.clearProgress();
      });

      expect(localStorageMock.removeItem).toHaveBeenCalledWith('brain_config_wizard_progress');
    });

    it('should handle localStorage errors gracefully', () => {
      localStorageMock.setItem.mockImplementation(() => {
        throw new Error('Storage full');
      });

      const { result } = renderHook(() => useConfigWizard());

      // Should not throw
      expect(() => {
        act(() => {
          result.current.saveProgress();
        });
      }).not.toThrow();
    });
  });

  describe('Step Status', () => {
    it('should return correct step status', async () => {
      const { result } = renderHook(() => useConfigWizard());

      // Validate steps to mark them as completed
      await act(async () => {
        result.current.updateConfig({
          appName: 'test',
          baseUrl: 'http://localhost'
        });
        await result.current.validateStep(0);
      });

      await act(async () => {
        result.current.updateConfig({
          database: {
            type: 'sqlite',
            path: './test.db'
          }
        });
        await result.current.validateStep(1);
      });

      act(() => {
        result.current.setCurrentStep(2);
      });

      expect(result.current.completedSteps).toContain(0);
      expect(result.current.completedSteps).toContain(1);
      expect(result.current.getStepStatus(0)).toBe('completed');
      expect(result.current.getStepStatus(1)).toBe('completed');
      expect(result.current.getStepStatus(2)).toBe('current');
      expect(result.current.getStepStatus(3)).toBe('upcoming');
    });
  });

  describe('Validate All Steps', () => {
    it('should validate all steps sequentially', async () => {
      const { result } = renderHook(() => useConfigWizard());

      // Set up valid configuration
      act(() => {
        result.current.updateConfig({
          appName: 'test',
          baseUrl: 'http://localhost',
          database: {
            type: 'postgresql',
            host: 'localhost',
            name: 'test_db',
            username: 'admin',
            password: 'password'
          },
          apiKeys: {
            openai: 'sk-test',
            supabase_url: 'https://test.supabase.co',
            supabase_key: 'test-key'
          }
        });
      });

      let allValid: boolean = false;
      await act(async () => {
        allValid = await result.current.validateAllSteps();
      });

      expect(allValid).toBe(true);
    });

    it('should stop validation on first error', async () => {
      const { result } = renderHook(() => useConfigWizard());

      // Invalid configuration (missing app name)
      act(() => {
        result.current.updateConfig({ appName: '' });
      });

      let allValid: boolean = false;
      await act(async () => {
        allValid = await result.current.validateAllSteps();
      });

      expect(allValid).toBe(false);
    });
  });
});