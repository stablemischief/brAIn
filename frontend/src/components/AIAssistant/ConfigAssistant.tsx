import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, HelpCircle, Sparkles } from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
}

interface ConfigAssistantProps {
  currentStep?: string;
  config?: any;
  onSuggestion?: (suggestion: any) => void;
}

const SAMPLE_SUGGESTIONS = [
  "What's the recommended database configuration for production?",
  "How do I set up SSL for my database connection?",
  "What API keys are required vs optional?",
  "Can you help me test my configuration?",
  "What services should I enable for my use case?"
];

export const ConfigAssistant: React.FC<ConfigAssistantProps> = ({
  currentStep,
  config,
  onSuggestion
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "👋 Hi! I'm your AI Configuration Assistant. I can help you set up your brAIn system. Feel free to ask me anything about the configuration process!",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const generateResponse = async (userMessage: string): Promise<string> => {
    // Simulate AI response generation
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Context-aware responses based on current step
    if (userMessage.toLowerCase().includes('database')) {
      return "For production environments, I recommend using PostgreSQL with connection pooling enabled. Set the pool size to 10-20 connections for most applications. Don't forget to enable SSL for secure connections!";
    }
    if (userMessage.toLowerCase().includes('api key')) {
      return "Required API keys are OpenAI (for AI features) and Supabase (for database). Optional keys include Anthropic (for Claude AI), Google Drive (for file integration), and Langfuse (for monitoring). Make sure to keep your keys secure!";
    }
    if (userMessage.toLowerCase().includes('ssl')) {
      return "To enable SSL for your database: 1) Toggle the 'Enable SSL Connection' option in Advanced Settings, 2) Ensure your database server has SSL certificates configured, 3) For PostgreSQL, you may need to add '?sslmode=require' to your connection string.";
    }
    if (userMessage.toLowerCase().includes('test')) {
      return "You can test your configuration in the Review step. Click 'Run All Tests' to validate your environment, database connection, API keys, and service dependencies. All tests should pass before completing setup.";
    }
    if (userMessage.toLowerCase().includes('service')) {
      return "Essential services to enable: Real-time Updates (for live dashboard), System Monitoring (for health checks), and Cost Tracking (to monitor API usage). Knowledge Graph is great for document relationships. Enable Auto Backups for production!";
    }

    // Default response
    return `I understand you're asking about "${userMessage}". Based on your current configuration step, I recommend checking the documentation for detailed guidance. Is there a specific aspect you'd like help with?`;
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Generate AI response
    const response = await generateResponse(input);

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response,
      timestamp: new Date()
    };

    setIsTyping(false);
    setMessages(prev => [...prev, assistantMessage]);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-2">
          <div className="relative">
            <Bot className="w-8 h-8 text-blue-500" />
            <Sparkles className="w-3 h-3 text-yellow-400 absolute -top-1 -right-1" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-gray-100">
              AI Configuration Assistant
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Powered by Pydantic AI
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex items-start space-x-2 max-w-[80%] ${
              message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
            }`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
              }`}>
                {message.role === 'user' ? (
                  <User className="w-4 h-4" />
                ) : (
                  <Bot className="w-4 h-4" />
                )}
              </div>
              <div className={`rounded-lg px-4 py-2 ${
                message.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
              }`}>
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                <p className={`text-xs mt-1 ${
                  message.role === 'user'
                    ? 'text-blue-100'
                    : 'text-gray-500 dark:text-gray-400'
                }`}>
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex justify-start">
            <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-700 rounded-lg px-4 py-2">
              <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
              <span className="text-sm text-gray-600 dark:text-gray-400">AI is thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions */}
      {messages.length <= 2 && (
        <div className="px-4 pb-2">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
            Quick questions:
          </p>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_SUGGESTIONS.slice(0, 3).map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className="text-xs px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700
                         dark:text-gray-300 rounded-full hover:bg-gray-200
                         dark:hover:bg-gray-600 transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask me anything about configuration..."
            className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                     focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isTyping}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600
                     disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
          <HelpCircle className="w-3 h-3 inline mr-1" />
          AI assistance is context-aware based on your current configuration step
        </p>
      </div>
    </div>
  );
};