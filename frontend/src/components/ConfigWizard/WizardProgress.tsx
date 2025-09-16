import React from 'react';
import { Check, AlertCircle } from 'lucide-react';

interface WizardStep {
  id: string;
  label: string;
  icon: string;
}

interface WizardProgressProps {
  steps: WizardStep[];
  currentStep: number;
  errors: Record<string, string>;
  completedSteps?: number[];
}

export const WizardProgress: React.FC<WizardProgressProps> = ({
  steps,
  currentStep,
  errors,
  completedSteps = []
}) => {
  const getStepStatus = (index: number) => {
    if (completedSteps.includes(index)) return 'completed';
    if (index === currentStep) return 'current';
    if (index < currentStep) return 'completed';
    return 'upcoming';
  };

  const getStepClassName = (status: string) => {
    const base = 'relative flex items-center justify-center w-10 h-10 rounded-full transition-all';

    switch (status) {
      case 'completed':
        return `${base} bg-green-500 text-white`;
      case 'current':
        return `${base} bg-blue-500 text-white ring-4 ring-blue-200 dark:ring-blue-800`;
      case 'upcoming':
        return `${base} bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400`;
      default:
        return base;
    }
  };

  return (
    <div className="relative">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const status = getStepStatus(index);
          const hasError = index === currentStep && Object.keys(errors).length > 0;

          return (
            <React.Fragment key={step.id}>
              <div className="flex flex-col items-center">
                <div className={getStepClassName(status)}>
                  {status === 'completed' ? (
                    <Check className="w-5 h-5" />
                  ) : hasError ? (
                    <AlertCircle className="w-5 h-5" />
                  ) : (
                    <span className="text-sm font-semibold">{index + 1}</span>
                  )}
                </div>
                <div className="mt-2 text-center">
                  <p className={`text-xs font-medium ${
                    status === 'current'
                      ? 'text-blue-600 dark:text-blue-400'
                      : status === 'completed'
                      ? 'text-green-600 dark:text-green-400'
                      : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    {step.label}
                  </p>
                </div>
              </div>

              {index < steps.length - 1 && (
                <div className="flex-1 h-0.5 mx-2">
                  <div className="h-full bg-gray-300 dark:bg-gray-600 relative">
                    <div
                      className="absolute h-full bg-green-500 transition-all duration-300"
                      style={{
                        width: index < currentStep ? '100%' : '0%'
                      }}
                    />
                  </div>
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};