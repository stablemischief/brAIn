import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Check, AlertCircle } from 'lucide-react';
import { WizardProgress } from './WizardProgress';
import { WizardNavigation } from './WizardNavigation';
import { useConfigWizard } from '../../hooks/useConfigWizard';
import { ConfigTemplate } from '../../types';

// Import step components
import { EnvironmentStep } from '../ConfigSteps/EnvironmentStep';
import { DatabaseStep } from '../ConfigSteps/DatabaseStep';
import { APIKeysStep } from '../ConfigSteps/APIKeysStep';
import { ServicesStep } from '../ConfigSteps/ServicesStep';
import { ReviewStep } from '../ConfigSteps/ReviewStep';

interface ConfigWizardProps {
  onComplete?: (config: any) => void;
  onCancel?: () => void;
  initialConfig?: Partial<any>;
}

const WIZARD_STEPS = [
  { id: 'environment', label: 'Environment', icon: '🌍' },
  { id: 'database', label: 'Database', icon: '🗄️' },
  { id: 'api-keys', label: 'API Keys', icon: '🔑' },
  { id: 'services', label: 'Services', icon: '⚙️' },
  { id: 'review', label: 'Review & Test', icon: '✅' }
];

export const ConfigWizard: React.FC<ConfigWizardProps> = ({
  onComplete,
  onCancel,
  initialConfig
}) => {
  const {
    currentStep,
    config,
    errors,
    isValidating,
    setCurrentStep,
    updateConfig,
    validateStep,
    saveProgress,
    loadProgress,
    clearProgress
  } = useConfigWizard(initialConfig);

  const [showAIAssistant, setShowAIAssistant] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<ConfigTemplate | null>(null);

  // Load saved progress on mount
  useEffect(() => {
    const savedProgress = loadProgress();
    if (savedProgress && !initialConfig) {
      // Optionally show a toast or modal asking if user wants to resume
      console.log('Found saved progress:', savedProgress);
    }
  }, []);

  // Auto-save progress on config changes
  useEffect(() => {
    const saveTimer = setTimeout(() => {
      saveProgress();
    }, 1000);
    return () => clearTimeout(saveTimer);
  }, [config, currentStep]);

  const handleNext = async () => {
    const isValid = await validateStep(currentStep);
    if (isValid && currentStep < WIZARD_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleComplete = async () => {
    const isValid = await validateStep(currentStep);
    if (isValid) {
      clearProgress();
      onComplete?.(config);
    }
  };

  const handleTemplateSelect = (template: ConfigTemplate) => {
    setSelectedTemplate(template);
    updateConfig(template.config);
  };

  const renderStepContent = () => {
    const stepId = WIZARD_STEPS[currentStep].id;
    const commonProps = {
      config,
      errors,
      onUpdate: updateConfig,
      isValidating
    };

    switch (stepId) {
      case 'environment':
        return <EnvironmentStep {...commonProps} />;
      case 'database':
        return <DatabaseStep {...commonProps} />;
      case 'api-keys':
        return <APIKeysStep {...commonProps} />;
      case 'services':
        return <ServicesStep {...commonProps} />;
      case 'review':
        return <ReviewStep {...commonProps} onTest={handleComplete} />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-5xl mx-auto px-4">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Configuration Wizard
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Let's set up your brAIn system step by step
          </p>
        </div>

        {/* Progress Bar */}
        <WizardProgress
          steps={WIZARD_STEPS}
          currentStep={currentStep}
          errors={errors}
        />

        {/* Main Content Area */}
        <div className="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow-lg">
          <div className="p-8">
            {/* Step Header */}
            <div className="mb-6 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-3xl">{WIZARD_STEPS[currentStep].icon}</span>
                <div>
                  <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
                    {WIZARD_STEPS[currentStep].label}
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Step {currentStep + 1} of {WIZARD_STEPS.length}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowAIAssistant(!showAIAssistant)}
                className="px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-700
                         dark:text-blue-400 dark:hover:text-blue-300 transition-colors"
              >
                {showAIAssistant ? 'Hide' : 'Show'} AI Assistant
              </button>
            </div>

            {/* Step Content */}
            <div className="min-h-[400px]">
              {renderStepContent()}
            </div>

            {/* Error Summary */}
            {Object.keys(errors).length > 0 && (
              <div className="mt-6 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                <div className="flex items-start space-x-2">
                  <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 mt-0.5" />
                  <div>
                    <p className="font-medium text-red-800 dark:text-red-200">
                      Please fix the following errors:
                    </p>
                    <ul className="mt-2 space-y-1 text-sm text-red-700 dark:text-red-300">
                      {Object.entries(errors).map(([field, error]) => (
                        <li key={field}>• {error}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Navigation */}
          <WizardNavigation
            currentStep={currentStep}
            totalSteps={WIZARD_STEPS.length}
            onPrevious={handlePrevious}
            onNext={handleNext}
            onComplete={handleComplete}
            onCancel={onCancel}
            isValidating={isValidating}
            hasErrors={Object.keys(errors).length > 0}
          />
        </div>

        {/* Template Selection (Optional) */}
        {currentStep === 0 && (
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              💡 <strong>Tip:</strong> You can use a template to quickly configure common scenarios.
              <button
                className="ml-2 underline hover:no-underline"
                onClick={() => {/* TODO: Show template modal */}}
              >
                Browse Templates
              </button>
            </p>
          </div>
        )}
      </div>

      {/* AI Assistant Panel (Slide-in from right) */}
      {showAIAssistant && (
        <div className="fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-800 shadow-2xl
                      transform transition-transform duration-300 z-50">
          <div className="p-6 h-full flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                AI Configuration Assistant
              </h3>
              <button
                onClick={() => setShowAIAssistant(false)}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400
                         dark:hover:text-gray-200"
              >
                ✕
              </button>
            </div>
            {/* AI Assistant component will be implemented separately */}
            <div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <p className="text-gray-600 dark:text-gray-400 text-center mt-8">
                AI Assistant coming soon...
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};