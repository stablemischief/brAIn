"""
Token counting utilities for accurate cost calculation across different models
Supports OpenAI, Claude, Gemini, and other LLM providers
"""

import logging
import tiktoken
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


@dataclass
class TokenCount:
    """Token count result with breakdown"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    provider: ModelProvider
    encoding_name: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    provider: ModelProvider
    encoding: str
    max_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    supports_functions: bool = True
    context_window: int = 4096


class TokenCounter:
    """
    Accurate token counting for multiple LLM providers and models
    """
    
    def __init__(self):
        self.encodings = {}
        self.model_configs = self._initialize_model_configs()
        
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize model configurations with current pricing"""
        return {
            # OpenAI Models
            "gpt-4": ModelConfig(
                name="gpt-4",
                provider=ModelProvider.OPENAI,
                encoding="cl100k_base",
                max_tokens=8192,
                input_cost_per_1k=0.03,
                output_cost_per_1k=0.06,
                context_window=8192
            ),
            "gpt-4-turbo": ModelConfig(
                name="gpt-4-turbo",
                provider=ModelProvider.OPENAI,
                encoding="cl100k_base",
                max_tokens=4096,
                input_cost_per_1k=0.01,
                output_cost_per_1k=0.03,
                context_window=128000
            ),
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                provider=ModelProvider.OPENAI,
                encoding="cl100k_base",
                max_tokens=4096,
                input_cost_per_1k=0.005,
                output_cost_per_1k=0.015,
                context_window=128000
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini",
                provider=ModelProvider.OPENAI,
                encoding="cl100k_base",
                max_tokens=16384,
                input_cost_per_1k=0.00015,
                output_cost_per_1k=0.0006,
                context_window=128000
            ),
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                encoding="cl100k_base",
                max_tokens=4096,
                input_cost_per_1k=0.0015,
                output_cost_per_1k=0.002,
                context_window=16385
            ),
            # Embedding Models
            "text-embedding-ada-002": ModelConfig(
                name="text-embedding-ada-002",
                provider=ModelProvider.OPENAI,
                encoding="cl100k_base",
                max_tokens=8191,
                input_cost_per_1k=0.0001,
                output_cost_per_1k=0.0,  # Embeddings don't have output cost
                context_window=8191,
                supports_functions=False
            ),
            "text-embedding-3-small": ModelConfig(
                name="text-embedding-3-small",
                provider=ModelProvider.OPENAI,
                encoding="cl100k_base",
                max_tokens=8191,
                input_cost_per_1k=0.00002,
                output_cost_per_1k=0.0,
                context_window=8191,
                supports_functions=False
            ),
            "text-embedding-3-large": ModelConfig(
                name="text-embedding-3-large",
                provider=ModelProvider.OPENAI,
                encoding="cl100k_base",
                max_tokens=8191,
                input_cost_per_1k=0.00013,
                output_cost_per_1k=0.0,
                context_window=8191,
                supports_functions=False
            ),
            # Anthropic Models (estimated pricing)
            "claude-3-haiku": ModelConfig(
                name="claude-3-haiku",
                provider=ModelProvider.ANTHROPIC,
                encoding="cl100k_base",  # Approximation
                max_tokens=4096,
                input_cost_per_1k=0.00025,
                output_cost_per_1k=0.00125,
                context_window=200000
            ),
            "claude-3-sonnet": ModelConfig(
                name="claude-3-sonnet",
                provider=ModelProvider.ANTHROPIC,
                encoding="cl100k_base",  # Approximation
                max_tokens=4096,
                input_cost_per_1k=0.003,
                output_cost_per_1k=0.015,
                context_window=200000
            ),
            "claude-3-opus": ModelConfig(
                name="claude-3-opus",
                provider=ModelProvider.ANTHROPIC,
                encoding="cl100k_base",  # Approximation
                max_tokens=4096,
                input_cost_per_1k=0.015,
                output_cost_per_1k=0.075,
                context_window=200000
            )
        }
    
    def get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get tiktoken encoding for a model"""
        config = self.model_configs.get(model)
        if not config:
            # Default to cl100k_base for unknown models
            encoding_name = "cl100k_base"
        else:
            encoding_name = config.encoding
            
        if encoding_name not in self.encodings:
            try:
                self.encodings[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to get encoding {encoding_name}: {e}")
                # Fallback to cl100k_base
                self.encodings[encoding_name] = tiktoken.get_encoding("cl100k_base")
                
        return self.encodings[encoding_name]
    
    def count_tokens_text(self, text: str, model: str = "gpt-4") -> TokenCount:
        """Count tokens in a text string"""
        encoding = self.get_encoding(model)
        token_count = len(encoding.encode(text))
        
        config = self.model_configs.get(model, self.model_configs["gpt-4"])
        
        return TokenCount(
            input_tokens=token_count,
            output_tokens=0,
            total_tokens=token_count,
            model=model,
            provider=config.provider,
            encoding_name=config.encoding
        )
    
    def count_tokens_messages(
        self, 
        messages: List[Dict], 
        model: str = "gpt-4",
        functions: Optional[List[Dict]] = None
    ) -> TokenCount:
        """
        Count tokens in a list of messages (OpenAI chat format)
        Accounts for message structure overhead and function calls
        """
        encoding = self.get_encoding(model)
        config = self.model_configs.get(model, self.model_configs["gpt-4"])
        
        # Base tokens per message and name
        if model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        elif model.startswith("gpt-3.5-turbo"):
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            tokens_per_message = 3
            tokens_per_name = 1
        
        total_tokens = 0
        
        # Count message tokens
        for message in messages:
            total_tokens += tokens_per_message
            
            for key, value in message.items():
                if key == "name":
                    total_tokens += tokens_per_name
                
                if isinstance(value, str):
                    total_tokens += len(encoding.encode(value))
                elif isinstance(value, (list, dict)):
                    # Handle function calls, tool calls, etc.
                    total_tokens += len(encoding.encode(json.dumps(value)))
        
        # Add tokens for functions/tools if provided
        if functions and config.supports_functions:
            functions_text = json.dumps(functions)
            total_tokens += len(encoding.encode(functions_text))
        
        # Add completion priming tokens
        total_tokens += 3
        
        return TokenCount(
            input_tokens=total_tokens,
            output_tokens=0,
            total_tokens=total_tokens,
            model=model,
            provider=config.provider,
            encoding_name=config.encoding,
            metadata={
                "message_count": len(messages),
                "has_functions": bool(functions)
            }
        )
    
    def count_tokens_completion(
        self,
        prompt: str,
        completion: str,
        model: str = "gpt-4"
    ) -> TokenCount:
        """Count tokens for a prompt-completion pair"""
        encoding = self.get_encoding(model)
        config = self.model_configs.get(model, self.model_configs["gpt-4"])
        
        input_tokens = len(encoding.encode(prompt))
        output_tokens = len(encoding.encode(completion))
        
        return TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            model=model,
            provider=config.provider,
            encoding_name=config.encoding
        )
    
    def estimate_tokens_anthropic(self, text: str) -> int:
        """
        Estimate tokens for Anthropic models
        Claude uses a different tokenization, but we approximate with OpenAI's
        """
        # Anthropic tokens are roughly 1.2x OpenAI tokens for English text
        openai_tokens = len(self.get_encoding("cl100k_base").encode(text))
        return int(openai_tokens * 1.2)
    
    def estimate_tokens_google(self, text: str) -> int:
        """
        Estimate tokens for Google models (Gemini)
        Google uses SentencePiece tokenization
        """
        # Rough approximation: 1 token per 3-4 characters for English
        # This is a simplified estimation
        return max(1, len(text) // 3)
    
    def count_tokens_by_provider(
        self,
        text: str,
        model: str,
        provider: Optional[ModelProvider] = None
    ) -> TokenCount:
        """Count tokens using provider-specific methods"""
        if provider is None:
            config = self.model_configs.get(model)
            provider = config.provider if config else ModelProvider.OPENAI
        
        if provider == ModelProvider.ANTHROPIC:
            token_count = self.estimate_tokens_anthropic(text)
        elif provider == ModelProvider.GOOGLE:
            token_count = self.estimate_tokens_google(text)
        else:
            # Default to OpenAI tokenization
            token_count = len(self.get_encoding(model).encode(text))
        
        config = self.model_configs.get(model, self.model_configs["gpt-4"])
        
        return TokenCount(
            input_tokens=token_count,
            output_tokens=0,
            total_tokens=token_count,
            model=model,
            provider=provider,
            metadata={"estimation_method": provider.value}
        )
    
    def validate_context_length(
        self,
        token_count: TokenCount,
        model: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate that token count is within model's context window"""
        config = self.model_configs.get(model)
        if not config:
            return True, None  # Unknown model, assume valid
        
        if token_count.total_tokens > config.context_window:
            return False, f"Token count {token_count.total_tokens} exceeds model {model} context window of {config.context_window}"
        
        return True, None
    
    def get_model_info(self, model: str) -> Optional[ModelConfig]:
        """Get configuration information for a model"""
        return self.model_configs.get(model)
    
    def add_custom_model(self, config: ModelConfig):
        """Add a custom model configuration"""
        self.model_configs[config.name] = config
        logger.info(f"Added custom model configuration: {config.name}")
    
    def batch_count_tokens(
        self,
        texts: List[str],
        model: str = "gpt-4"
    ) -> List[TokenCount]:
        """Count tokens for a batch of texts efficiently"""
        results = []
        encoding = self.get_encoding(model)
        config = self.model_configs.get(model, self.model_configs["gpt-4"])
        
        for text in texts:
            token_count = len(encoding.encode(text))
            results.append(TokenCount(
                input_tokens=token_count,
                output_tokens=0,
                total_tokens=token_count,
                model=model,
                provider=config.provider,
                encoding_name=config.encoding
            ))
        
        return results
    
    def estimate_processing_tokens(
        self,
        document_content: str,
        processing_type: str,
        model: str = "gpt-4"
    ) -> TokenCount:
        """
        Estimate tokens needed for document processing
        Includes system prompts and processing overhead
        """
        base_tokens = self.count_tokens_text(document_content, model)
        
        # Processing overhead estimates
        overhead_multipliers = {
            "extraction": 1.2,  # System prompt + extraction format
            "summarization": 1.5,  # System prompt + summary generation
            "embedding": 1.0,  # Direct embedding, no overhead
            "classification": 1.3,  # System prompt + categories
            "entity_extraction": 1.4,  # System prompt + entity format
            "relationship_detection": 1.6,  # System prompt + relationship format
        }
        
        multiplier = overhead_multipliers.get(processing_type, 1.2)
        estimated_total = int(base_tokens.total_tokens * multiplier)
        
        # Add output tokens estimate for generation tasks
        output_estimate = 0
        if processing_type in ["summarization", "extraction", "classification"]:
            output_estimate = min(1000, base_tokens.total_tokens // 4)  # 25% of input, max 1000
        
        return TokenCount(
            input_tokens=estimated_total,
            output_tokens=output_estimate,
            total_tokens=estimated_total + output_estimate,
            model=model,
            provider=self.model_configs.get(model, self.model_configs["gpt-4"]).provider,
            metadata={
                "processing_type": processing_type,
                "overhead_multiplier": multiplier,
                "base_tokens": base_tokens.total_tokens
            }
        )


# Global token counter instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get the global token counter instance"""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


# Convenience functions
def count_tokens(text: str, model: str = "gpt-4") -> TokenCount:
    """Count tokens in text using the global counter"""
    return get_token_counter().count_tokens_text(text, model)


def count_message_tokens(messages: List[Dict], model: str = "gpt-4") -> TokenCount:
    """Count tokens in messages using the global counter"""
    return get_token_counter().count_tokens_messages(messages, model)


def estimate_processing_cost(
    document_content: str,
    processing_type: str,
    model: str = "gpt-4"
) -> Tuple[TokenCount, float]:
    """Estimate tokens and cost for document processing"""
    counter = get_token_counter()
    token_count = counter.estimate_processing_tokens(document_content, processing_type, model)
    
    config = counter.get_model_info(model)
    if not config:
        return token_count, 0.0
    
    input_cost = (token_count.input_tokens / 1000) * config.input_cost_per_1k
    output_cost = (token_count.output_tokens / 1000) * config.output_cost_per_1k
    total_cost = input_cost + output_cost
    
    return token_count, total_cost