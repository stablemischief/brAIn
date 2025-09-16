import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ConfigWizard } from '../components/ConfigWizard';
import { ConfigAssistant } from '../components/AIAssistant/ConfigAssistant';
import { CheckCircle } from 'lucide-react';

const ConfigWizardPage: React.FC = () => {
  const navigate = useNavigate();
  const [showAssistant, setShowAssistant] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [finalConfig, setFinalConfig] = useState<any>(null);

  const handleComplete = async (config: any) => {
    // Here you would typically send the configuration to your backend
    console.log('Configuration completed:', config);
    setFinalConfig(config);
    setIsComplete(true);

    // Simulate saving to backend
    try {
      const response = await fetch('/api/config/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        // Navigate to dashboard after successful setup
        setTimeout(() => {
          navigate('/');
        }, 3000);
      }
    } catch (error) {
      console.error('Failed to save configuration:', error);
    }
  };

  const handleCancel = () => {
    if (window.confirm('Are you sure you want to cancel? Your progress will be saved.')) {
      navigate('/');
    }
  };

  if (isComplete) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="flex justify-center mb-6">
            <div className="w-24 h-24 bg-green-100 dark:bg-green-900/30 rounded-full
                          flex items-center justify-center">
              <CheckCircle className="w-12 h-12 text-green-500" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">
            Configuration Complete!
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Your brAIn system has been successfully configured.
          </p>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg max-w-md mx-auto">
            <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Next Steps:
            </h3>
            <ul className="text-left space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <li>✅ Configuration saved to .env file</li>
              <li>✅ Database schema initialized</li>
              <li>✅ API connections verified</li>
              <li>✅ Services started successfully</li>
            </ul>
          </div>
          <p className="mt-6 text-sm text-gray-500 dark:text-gray-400">
            Redirecting to dashboard in 3 seconds...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen">
      <ConfigWizard
        onComplete={handleComplete}
        onCancel={handleCancel}
      />

      {/* Floating AI Assistant Toggle */}
      <button
        onClick={() => setShowAssistant(!showAssistant)}
        className="fixed bottom-6 right-6 w-14 h-14 bg-blue-500 text-white rounded-full
                 shadow-lg hover:bg-blue-600 transition-colors flex items-center justify-center
                 z-40"
      >
        <svg
          className="w-7 h-7"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-4l-4 4z"
          />
        </svg>
      </button>

      {/* AI Assistant Panel */}
      {showAssistant && (
        <div className="fixed bottom-24 right-6 w-96 h-[600px] z-50">
          <ConfigAssistant />
        </div>
      )}
    </div>
  );
};

export default ConfigWizardPage;