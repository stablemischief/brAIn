import React from 'react';
import { ChevronLeft, ChevronRight, Check, X, Loader2 } from 'lucide-react';

interface WizardNavigationProps {
  currentStep: number;
  totalSteps: number;
  onPrevious: () => void;
  onNext: () => void;
  onComplete: () => void;
  onCancel?: () => void;
  isValidating?: boolean;
  hasErrors?: boolean;
  disableNext?: boolean;
  disablePrevious?: boolean;
}

export const WizardNavigation: React.FC<WizardNavigationProps> = ({
  currentStep,
  totalSteps,
  onPrevious,
  onNext,
  onComplete,
  onCancel,
  isValidating = false,
  hasErrors = false,
  disableNext = false,
  disablePrevious = false
}) => {
  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === totalSteps - 1;

  return (
    <div className="px-8 py-4 bg-gray-50 dark:bg-gray-900 border-t border-gray-200
                  dark:border-gray-700 rounded-b-lg">
      <div className="flex items-center justify-between">
        {/* Cancel Button */}
        <button
          onClick={onCancel}
          className="flex items-center space-x-2 px-4 py-2 text-gray-600 dark:text-gray-400
                   hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
        >
          <X className="w-4 h-4" />
          <span>Cancel</span>
        </button>

        {/* Navigation Buttons */}
        <div className="flex items-center space-x-3">
          {/* Previous Button */}
          {!isFirstStep && (
            <button
              onClick={onPrevious}
              disabled={disablePrevious || isValidating}
              className="flex items-center space-x-2 px-4 py-2 text-gray-700 dark:text-gray-300
                       bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600
                       rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              <span>Previous</span>
            </button>
          )}

          {/* Next/Complete Button */}
          {isLastStep ? (
            <button
              onClick={onComplete}
              disabled={hasErrors || isValidating}
              className="flex items-center space-x-2 px-6 py-2 bg-green-600 text-white
                       rounded-lg hover:bg-green-700 disabled:opacity-50
                       disabled:cursor-not-allowed transition-colors"
            >
              {isValidating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Validating...</span>
                </>
              ) : (
                <>
                  <Check className="w-4 h-4" />
                  <span>Complete Setup</span>
                </>
              )}
            </button>
          ) : (
            <button
              onClick={onNext}
              disabled={disableNext || hasErrors || isValidating}
              className="flex items-center space-x-2 px-6 py-2 bg-blue-600 text-white
                       rounded-lg hover:bg-blue-700 disabled:opacity-50
                       disabled:cursor-not-allowed transition-colors"
            >
              {isValidating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Validating...</span>
                </>
              ) : (
                <>
                  <span>Next</span>
                  <ChevronRight className="w-4 h-4" />
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};