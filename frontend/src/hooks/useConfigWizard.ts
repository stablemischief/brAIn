import { useState, useEffect, useCallback } from 'react';

interface ConfigWizardState {
  currentStep: number;
  config: any;
  errors: Record<string, string>;
  isValidating: boolean;
  completedSteps: number[];
}

const STORAGE_KEY = 'brain_config_wizard_progress';

export const useConfigWizard = (initialConfig?: any) => {
  const [state, setState] = useState<ConfigWizardState>({
    currentStep: 0,
    config: initialConfig || {
      environment: 'development',
      appName: '',
      backendPort: 8000,
      frontendPort: 3000,
      baseUrl: 'http://localhost:3000',
      debugMode: false,
      database: {
        type: 'postgresql',
        host: 'localhost',
        port: 5432,
        name: '',
        username: '',
        password: '',
        ssl: false,
        poolSize: 10,
        timeout: 30000
      },
      apiKeys: {},
      services: []
    },
    errors: {},
    isValidating: false,
    completedSteps: []
  });

  // Load saved progress on mount
  const loadProgress = useCallback(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        return parsed;
      }
    } catch (error) {
      console.error('Failed to load saved progress:', error);
    }
    return null;
  }, []);

  // Save progress to localStorage
  const saveProgress = useCallback(() => {
    try {
      const toSave = {
        currentStep: state.currentStep,
        config: state.config,
        completedSteps: state.completedSteps,
        timestamp: Date.now()
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
    } catch (error) {
      console.error('Failed to save progress:', error);
    }
  }, [state]);

  // Clear saved progress
  const clearProgress = useCallback(() => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      console.error('Failed to clear progress:', error);
    }
  }, []);

  // Set current step
  const setCurrentStep = useCallback((step: number) => {
    setState(prev => ({ ...prev, currentStep: step }));
  }, []);

  // Update configuration
  const updateConfig = useCallback((updates: any) => {
    setState(prev => ({
      ...prev,
      config: { ...prev.config, ...updates }
    }));
  }, []);

  // Validate a specific step
  const validateStep = useCallback(async (step: number): Promise<boolean> => {
    setState(prev => ({ ...prev, isValidating: true, errors: {} }));

    const errors: Record<string, string> = {};

    // Step-specific validation
    switch (step) {
      case 0: // Environment Step
        if (!state.config.appName) {
          errors.appName = 'Application name is required';
        }
        if (!state.config.baseUrl) {
          errors.baseUrl = 'Base URL is required';
        }
        if (state.config.backendPort === state.config.frontendPort) {
          errors.backendPort = 'Backend and frontend ports must be different';
        }
        break;

      case 1: // Database Step
        if (state.config.database.type !== 'sqlite') {
          if (!state.config.database.host) {
            errors['database.host'] = 'Database host is required';
          }
          if (!state.config.database.name) {
            errors['database.name'] = 'Database name is required';
          }
          if (!state.config.database.username) {
            errors['database.username'] = 'Database username is required';
          }
        } else {
          if (!state.config.database.path) {
            errors['database.path'] = 'Database file path is required';
          }
        }
        break;

      case 2: // API Keys Step
        if (!state.config.apiKeys?.openai) {
          errors['apiKeys.openai'] = 'OpenAI API key is required';
        }
        if (!state.config.apiKeys?.supabase_url) {
          errors['apiKeys.supabase_url'] = 'Supabase URL is required';
        }
        if (!state.config.apiKeys?.supabase_key) {
          errors['apiKeys.supabase_key'] = 'Supabase key is required';
        }
        break;

      case 3: // Services Step
        // Services validation is optional
        break;

      case 4: // Review Step
        // Final validation happens here
        break;
    }

    // Simulate async validation
    await new Promise(resolve => setTimeout(resolve, 500));

    setState(prev => ({
      ...prev,
      isValidating: false,
      errors,
      completedSteps: Object.keys(errors).length === 0
        ? Array.from(new Set([...prev.completedSteps, step]))
        : prev.completedSteps
    }));

    return Object.keys(errors).length === 0;
  }, [state.config]);

  // Validate all steps
  const validateAllSteps = useCallback(async (): Promise<boolean> => {
    for (let i = 0; i < 5; i++) {
      const isValid = await validateStep(i);
      if (!isValid) return false;
    }
    return true;
  }, [validateStep]);

  // Get step status
  const getStepStatus = useCallback((stepIndex: number) => {
    if (state.completedSteps.includes(stepIndex)) return 'completed';
    if (stepIndex === state.currentStep) return 'current';
    if (stepIndex < state.currentStep) return 'visited';
    return 'upcoming';
  }, [state.completedSteps, state.currentStep]);

  return {
    currentStep: state.currentStep,
    config: state.config,
    errors: state.errors,
    isValidating: state.isValidating,
    completedSteps: state.completedSteps,
    setCurrentStep,
    updateConfig,
    validateStep,
    validateAllSteps,
    saveProgress,
    loadProgress,
    clearProgress,
    getStepStatus
  };
};