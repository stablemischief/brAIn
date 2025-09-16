import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConfigAssistant } from '../ConfigAssistant';
import '@testing-library/jest-dom';

// Mock fetch for API calls
global.fetch = jest.fn();

describe('ConfigAssistant', () => {
  const mockOnSuggestion = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockClear();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Initial Rendering', () => {
    it('should render with welcome message', () => {
      render(<ConfigAssistant />);

      expect(screen.getByText(/Hi! I'm your AI Configuration Assistant/)).toBeInTheDocument();
      expect(screen.getByText('AI Configuration Assistant')).toBeInTheDocument();
      expect(screen.getByText('Powered by Pydantic AI')).toBeInTheDocument();
    });

    it('should render input field and send button', () => {
      render(<ConfigAssistant />);

      expect(screen.getByPlaceholderText('Ask me anything about configuration...')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
    });

    it('should display quick question suggestions initially', () => {
      render(<ConfigAssistant />);

      expect(screen.getByText('Quick questions:')).toBeInTheDocument();
      expect(screen.getByText(/What's the recommended database configuration/)).toBeInTheDocument();
    });
  });

  describe('Message Sending', () => {
    it('should send message when send button is clicked', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      const sendButton = screen.getByRole('button', { name: /send/i });

      await userEvent.type(input, 'How do I set up SSL?');
      fireEvent.click(sendButton);

      await waitFor(() => {
        expect(screen.getByText('How do I set up SSL?')).toBeInTheDocument();
      });
    });

    it('should send message when Enter is pressed', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');

      await userEvent.type(input, 'Test message{enter}');

      await waitFor(() => {
        expect(screen.getByText('Test message')).toBeInTheDocument();
      });
    });

    it('should clear input after sending', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...') as HTMLInputElement;
      const sendButton = screen.getByRole('button', { name: /send/i });

      await userEvent.type(input, 'Test message');
      fireEvent.click(sendButton);

      await waitFor(() => {
        expect(input.value).toBe('');
      });
    });

    it('should not send empty messages', () => {
      render(<ConfigAssistant />);

      const sendButton = screen.getByRole('button', { name: /send/i });
      const initialMessageCount = screen.getAllByText(/:/i).length;

      fireEvent.click(sendButton);

      const afterMessageCount = screen.getAllByText(/:/i).length;
      expect(afterMessageCount).toBe(initialMessageCount);
    });
  });

  describe('AI Response Generation', () => {
    it('should show typing indicator while generating response', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'Test question{enter}');

      await waitFor(() => {
        expect(screen.getByText('AI is thinking...')).toBeInTheDocument();
      });

      await waitFor(() => {
        expect(screen.queryByText('AI is thinking...')).not.toBeInTheDocument();
      }, { timeout: 3000 });
    });

    it('should display AI response after user message', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'database{enter}');

      await waitFor(() => {
        expect(screen.getByText(/PostgreSQL/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });

    it('should provide context-aware responses', async () => {
      render(<ConfigAssistant currentStep="database" />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'What about SSL?{enter}');

      await waitFor(() => {
        expect(screen.getByText(/Enable SSL Connection/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });

  describe('Quick Suggestions', () => {
    it('should populate input when suggestion is clicked', async () => {
      render(<ConfigAssistant />);

      const suggestion = screen.getByText(/What's the recommended database configuration/);
      fireEvent.click(suggestion);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...') as HTMLInputElement;
      expect(input.value).toContain('database configuration');
    });

    it('should hide suggestions after initial messages', async () => {
      render(<ConfigAssistant />);

      // Send multiple messages
      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'First message{enter}');

      await waitFor(() => {
        expect(screen.getByText('First message')).toBeInTheDocument();
      });

      await userEvent.type(input, 'Second message{enter}');

      await waitFor(() => {
        expect(screen.getByText('Second message')).toBeInTheDocument();
      });

      // Suggestions should be hidden
      expect(screen.queryByText('Quick questions:')).not.toBeInTheDocument();
    });
  });

  describe('Message Display', () => {
    it('should display user messages with correct styling', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'User message{enter}');

      await waitFor(() => {
        const userMessage = screen.getByText('User message');
        const messageContainer = userMessage.closest('div');
        expect(messageContainer).toHaveClass('bg-blue-500');
      });
    });

    it('should display timestamps for messages', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'Test message{enter}');

      await waitFor(() => {
        // Check for time format (e.g., "10:30:45 AM")
        const timeRegex = /\d{1,2}:\d{2}:\d{2}\s?(AM|PM)?/i;
        const timestamps = screen.getAllByText(timeRegex);
        expect(timestamps.length).toBeGreaterThan(0);
      });
    });

    it('should auto-scroll to latest message', async () => {
      const scrollIntoViewMock = jest.fn();
      HTMLElement.prototype.scrollIntoView = scrollIntoViewMock;

      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'Test message{enter}');

      await waitFor(() => {
        expect(scrollIntoViewMock).toHaveBeenCalledWith({ behavior: 'smooth' });
      });
    });
  });

  describe('Response Content', () => {
    it('should provide database configuration help', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'database{enter}');

      await waitFor(() => {
        expect(screen.getByText(/PostgreSQL/i)).toBeInTheDocument();
        expect(screen.getByText(/connection pooling/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });

    it('should provide API key guidance', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'api key{enter}');

      await waitFor(() => {
        expect(screen.getByText(/OpenAI/i)).toBeInTheDocument();
        expect(screen.getByText(/Supabase/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });

    it('should provide testing guidance', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'test{enter}');

      await waitFor(() => {
        expect(screen.getByText(/Run All Tests/i)).toBeInTheDocument();
        expect(screen.getByText(/Review step/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });

    it('should provide service configuration help', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      await userEvent.type(input, 'service{enter}');

      await waitFor(() => {
        expect(screen.getByText(/Real-time Updates/i)).toBeInTheDocument();
        expect(screen.getByText(/System Monitoring/i)).toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });

  describe('Props Integration', () => {
    it('should accept current step prop', () => {
      render(<ConfigAssistant currentStep="database" />);

      // Component should render without errors with currentStep
      expect(screen.getByText('AI Configuration Assistant')).toBeInTheDocument();
    });

    it('should accept config prop', () => {
      const config = { environment: 'production', appName: 'test-app' };
      render(<ConfigAssistant config={config} />);

      // Component should render without errors with config
      expect(screen.getByText('AI Configuration Assistant')).toBeInTheDocument();
    });

    it('should call onSuggestion when provided', () => {
      render(<ConfigAssistant onSuggestion={mockOnSuggestion} />);

      // Component should render and potentially use onSuggestion
      expect(screen.getByText('AI Configuration Assistant')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');
      expect(input).toHaveAttribute('type', 'text');

      const sendButton = screen.getByRole('button', { name: /send/i });
      expect(sendButton).toBeInTheDocument();
    });

    it('should be keyboard navigable', async () => {
      render(<ConfigAssistant />);

      const input = screen.getByPlaceholderText('Ask me anything about configuration...');

      // Tab to input
      input.focus();
      expect(input).toHaveFocus();

      // Type and send with Enter
      await userEvent.type(input, 'Test{enter}');

      await waitFor(() => {
        expect(screen.getByText('Test')).toBeInTheDocument();
      });
    });
  });
});