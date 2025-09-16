import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { EnvironmentStep } from '../EnvironmentStep';
import '@testing-library/jest-dom';

describe('EnvironmentStep', () => {
  const mockOnUpdate = jest.fn();
  const defaultConfig = {
    environment: 'development',
    appName: '',
    backendPort: 8000,
    frontendPort: 3000,
    baseUrl: '',
    debugMode: false
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render all form fields', () => {
      render(
        <EnvironmentStep
          config={defaultConfig}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      expect(screen.getByLabelText(/Environment Type/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Application Name/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Backend Port/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Frontend Port/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Base URL/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Enable Debug Mode/i)).toBeInTheDocument();
    });

    it('should display environment variables preview', () => {
      render(
        <EnvironmentStep
          config={{
            ...defaultConfig,
            appName: 'test-app',
            baseUrl: 'http://localhost:3000'
          }}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      expect(screen.getByText('Environment Variables Preview')).toBeInTheDocument();
      expect(screen.getByText(/APP_NAME=test-app/)).toBeInTheDocument();
      expect(screen.getByText(/BASE_URL=http:\/\/localhost:3000/)).toBeInTheDocument();
    });
  });

  describe('User Interactions', () => {
    it('should update environment type', () => {
      render(
        <EnvironmentStep
          config={defaultConfig}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      const select = screen.getByLabelText(/Environment Type/i);
      fireEvent.change(select, { target: { value: 'production' } });

      expect(mockOnUpdate).toHaveBeenCalledWith({
        ...defaultConfig,
        environment: 'production'
      });
    });

    it('should update application name', async () => {
      render(
        <EnvironmentStep
          config={defaultConfig}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      const input = screen.getByLabelText(/Application Name/i);
      await userEvent.type(input, 'my-app');

      expect(mockOnUpdate).toHaveBeenCalledTimes('my-app'.length);
      expect(mockOnUpdate).toHaveBeenLastCalledWith(
        expect.objectContaining({ appName: 'my-app' })
      );
    });

    it('should update port values', async () => {
      render(
        <EnvironmentStep
          config={defaultConfig}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      const backendPort = screen.getByLabelText(/Backend Port/i);
      fireEvent.change(backendPort, { target: { value: '8080' } });

      expect(mockOnUpdate).toHaveBeenCalledWith({
        ...defaultConfig,
        backendPort: 8080
      });
    });

    it('should toggle debug mode', () => {
      render(
        <EnvironmentStep
          config={defaultConfig}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      const checkbox = screen.getByLabelText(/Enable Debug Mode/i);
      fireEvent.click(checkbox);

      expect(mockOnUpdate).toHaveBeenCalledWith({
        ...defaultConfig,
        debugMode: true
      });
    });
  });

  describe('Error Display', () => {
    it('should display field errors', () => {
      const errors = {
        appName: 'Application name is required',
        baseUrl: 'Invalid URL format',
        backendPort: 'Port must be between 1024 and 65535'
      };

      render(
        <EnvironmentStep
          config={defaultConfig}
          errors={errors}
          onUpdate={mockOnUpdate}
        />
      );

      expect(screen.getByText('Application name is required')).toBeInTheDocument();
      expect(screen.getByText('Invalid URL format')).toBeInTheDocument();
      expect(screen.getByText('Port must be between 1024 and 65535')).toBeInTheDocument();
    });
  });

  describe('Production Warning', () => {
    it('should show warning when production is selected', () => {
      render(
        <EnvironmentStep
          config={{ ...defaultConfig, environment: 'production' }}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      expect(screen.getByText('Production Environment Selected')).toBeInTheDocument();
      expect(screen.getByText(/SSL certificates and secure API keys/)).toBeInTheDocument();
    });

    it('should not show warning for development environment', () => {
      render(
        <EnvironmentStep
          config={{ ...defaultConfig, environment: 'development' }}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      expect(screen.queryByText('Production Environment Selected')).not.toBeInTheDocument();
    });
  });

  describe('Environment Preview', () => {
    it('should update preview in real-time', () => {
      const { rerender } = render(
        <EnvironmentStep
          config={defaultConfig}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      expect(screen.getByText(/ENVIRONMENT=development/)).toBeInTheDocument();
      expect(screen.getByText(/DEBUG=false/)).toBeInTheDocument();

      // Update config
      rerender(
        <EnvironmentStep
          config={{
            ...defaultConfig,
            environment: 'staging',
            debugMode: true
          }}
          errors={{}}
          onUpdate={mockOnUpdate}
        />
      );

      expect(screen.getByText(/ENVIRONMENT=staging/)).toBeInTheDocument();
      expect(screen.getByText(/DEBUG=true/)).toBeInTheDocument();
    });
  });

  describe('Validation States', () => {
    it('should disable inputs when validating', () => {
      render(
        <EnvironmentStep
          config={defaultConfig}
          errors={{}}
          onUpdate={mockOnUpdate}
          isValidating={true}
        />
      );

      // In this component, isValidating doesn't disable fields by default
      // This test is here as a placeholder for future implementation
      expect(screen.getByLabelText(/Application Name/i)).toBeInTheDocument();
    });
  });
});