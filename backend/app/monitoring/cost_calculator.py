"""
brAIn v2.0 Cost Calculator
Accurate token counting and cost calculation for LLM operations.
"""

import re
import logging
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
import tiktoken

logger = logging.getLogger(__name__)


class ModelPricing(str, Enum):
    """Supported models with their pricing information"""

    # OpenAI Models
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"

    # Embedding Models
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    # Claude Models (Anthropic via OpenAI-compatible API)
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"


class TokenUsage(BaseModel):
    """Token usage statistics for an LLM operation"""

    input_tokens: int = Field(default=0, description="Number of input tokens")

    output_tokens: int = Field(default=0, description="Number of output tokens")

    @property
    def total_tokens(self) -> int:
        """Total tokens used"""
        return self.input_tokens + self.output_tokens


class ModelConfig(BaseModel):
    """Configuration for a specific model"""

    name: str = Field(description="Model name")

    input_price_per_1k: float = Field(description="Price per 1K input tokens in USD")

    output_price_per_1k: float = Field(description="Price per 1K output tokens in USD")

    max_tokens: int = Field(description="Maximum tokens supported by the model")

    encoding_name: str = Field(
        default="cl100k_base", description="Tiktoken encoding name for token counting"
    )

    supports_embeddings: bool = Field(
        default=False, description="Whether this is an embedding model"
    )


class CostCalculator:
    """
    Calculates costs and token usage for LLM operations.

    Features:
    - Accurate token counting using tiktoken
    - Up-to-date pricing for all supported models
    - Batch cost calculation
    - Token estimation for planning
    """

    # Current pricing as of 2025 (USD per 1K tokens)
    MODEL_CONFIGS: Dict[str, ModelConfig] = {
        # OpenAI GPT-4 Models
        ModelPricing.GPT_4_TURBO.value: ModelConfig(
            name="GPT-4 Turbo",
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
            max_tokens=128000,
            encoding_name="cl100k_base",
        ),
        ModelPricing.GPT_4_TURBO_PREVIEW.value: ModelConfig(
            name="GPT-4 Turbo Preview",
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
            max_tokens=128000,
            encoding_name="cl100k_base",
        ),
        ModelPricing.GPT_4.value: ModelConfig(
            name="GPT-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            max_tokens=8192,
            encoding_name="cl100k_base",
        ),
        ModelPricing.GPT_4_32K.value: ModelConfig(
            name="GPT-4 32K",
            input_price_per_1k=0.06,
            output_price_per_1k=0.12,
            max_tokens=32768,
            encoding_name="cl100k_base",
        ),
        ModelPricing.GPT_3_5_TURBO.value: ModelConfig(
            name="GPT-3.5 Turbo",
            input_price_per_1k=0.001,
            output_price_per_1k=0.002,
            max_tokens=16385,
            encoding_name="cl100k_base",
        ),
        ModelPricing.GPT_3_5_TURBO_16K.value: ModelConfig(
            name="GPT-3.5 Turbo 16K",
            input_price_per_1k=0.003,
            output_price_per_1k=0.004,
            max_tokens=16385,
            encoding_name="cl100k_base",
        ),
        # OpenAI Embedding Models
        ModelPricing.TEXT_EMBEDDING_3_SMALL.value: ModelConfig(
            name="Text Embedding 3 Small",
            input_price_per_1k=0.00002,
            output_price_per_1k=0.0,  # No output tokens for embeddings
            max_tokens=8191,
            encoding_name="cl100k_base",
            supports_embeddings=True,
        ),
        ModelPricing.TEXT_EMBEDDING_3_LARGE.value: ModelConfig(
            name="Text Embedding 3 Large",
            input_price_per_1k=0.00013,
            output_price_per_1k=0.0,
            max_tokens=8191,
            encoding_name="cl100k_base",
            supports_embeddings=True,
        ),
        ModelPricing.TEXT_EMBEDDING_ADA_002.value: ModelConfig(
            name="Text Embedding Ada 002",
            input_price_per_1k=0.0001,
            output_price_per_1k=0.0,
            max_tokens=8191,
            encoding_name="cl100k_base",
            supports_embeddings=True,
        ),
        # Claude Models (Anthropic pricing)
        ModelPricing.CLAUDE_3_OPUS.value: ModelConfig(
            name="Claude 3 Opus",
            input_price_per_1k=0.015,
            output_price_per_1k=0.075,
            max_tokens=200000,
            encoding_name="cl100k_base",  # Approximation
        ),
        ModelPricing.CLAUDE_3_SONNET.value: ModelConfig(
            name="Claude 3 Sonnet",
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
            max_tokens=200000,
            encoding_name="cl100k_base",  # Approximation
        ),
        ModelPricing.CLAUDE_3_HAIKU.value: ModelConfig(
            name="Claude 3 Haiku",
            input_price_per_1k=0.00025,
            output_price_per_1k=0.00125,
            max_tokens=200000,
            encoding_name="cl100k_base",  # Approximation
        ),
        ModelPricing.CLAUDE_SONNET_4.value: ModelConfig(
            name="Claude Sonnet 4",
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
            max_tokens=200000,
            encoding_name="cl100k_base",  # Approximation
        ),
    }

    def __init__(self):
        self._encoders: Dict[str, Any] = {}

    def _get_encoder(self, encoding_name: str) -> Any:
        """Get or create tiktoken encoder for the given encoding"""
        if encoding_name not in self._encoders:
            try:
                self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load encoding {encoding_name}: {e}")
                # Fallback to cl100k_base
                self._encoders[encoding_name] = tiktoken.get_encoding("cl100k_base")

        return self._encoders[encoding_name]

    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens in text for a specific model.

        Args:
            text: Text to count tokens for
            model_name: Name of the model

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        # Get model config
        config = self.MODEL_CONFIGS.get(model_name)
        if not config:
            # Use default encoding for unknown models
            encoding_name = "cl100k_base"
            logger.warning(f"Unknown model {model_name}, using default encoding")
        else:
            encoding_name = config.encoding_name

        try:
            encoder = self._get_encoder(encoding_name)
            tokens = encoder.encode(str(text))
            return len(tokens)

        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(str(text)) // 4

    def estimate_tokens(
        self,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        model_name: str = "gpt-4-turbo",
    ) -> TokenUsage:
        """
        Estimate token usage for input and output text.

        Args:
            input_text: Input text
            output_text: Output text
            model_name: Model name for token counting

        Returns:
            TokenUsage object
        """
        input_tokens = self.count_tokens(input_text or "", model_name)
        output_tokens = self.count_tokens(output_text or "", model_name)

        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def calculate_cost(self, model_name: str, token_usage: TokenUsage) -> float:
        """
        Calculate cost for a given token usage.

        Args:
            model_name: Name of the model used
            token_usage: Token usage statistics

        Returns:
            Cost in USD
        """
        config = self.MODEL_CONFIGS.get(model_name)
        if not config:
            logger.warning(f"Unknown model {model_name}, cannot calculate cost")
            return 0.0

        # Calculate input cost
        input_cost = (token_usage.input_tokens / 1000.0) * config.input_price_per_1k

        # Calculate output cost (only for non-embedding models)
        output_cost = 0.0
        if not config.supports_embeddings:
            output_cost = (
                token_usage.output_tokens / 1000.0
            ) * config.output_price_per_1k

        total_cost = input_cost + output_cost

        return round(total_cost, 6)  # Round to 6 decimal places for precision

    def calculate_batch_cost(
        self, operations: list[Tuple[str, TokenUsage]]
    ) -> Dict[str, Any]:
        """
        Calculate costs for multiple operations.

        Args:
            operations: List of (model_name, token_usage) tuples

        Returns:
            Dictionary with cost breakdown
        """
        total_cost = 0.0
        model_costs = {}
        model_tokens = {}

        for model_name, token_usage in operations:
            cost = self.calculate_cost(model_name, token_usage)
            total_cost += cost

            if model_name not in model_costs:
                model_costs[model_name] = 0.0
                model_tokens[model_name] = TokenUsage()

            model_costs[model_name] += cost
            model_tokens[model_name].input_tokens += token_usage.input_tokens
            model_tokens[model_name].output_tokens += token_usage.output_tokens

        return {
            "total_cost": round(total_cost, 6),
            "model_breakdown": {
                model: {
                    "cost": round(cost, 6),
                    "tokens": model_tokens[model].model_dump(),
                }
                for model, cost in model_costs.items()
            },
            "operation_count": len(operations),
        }

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model information or None if not found
        """
        config = self.MODEL_CONFIGS.get(model_name)
        if not config:
            return None

        return {
            "name": config.name,
            "input_price_per_1k": config.input_price_per_1k,
            "output_price_per_1k": config.output_price_per_1k,
            "max_tokens": config.max_tokens,
            "supports_embeddings": config.supports_embeddings,
        }

    def list_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all supported models with their information.

        Returns:
            Dictionary of model information
        """
        return {
            model_name: {
                "name": config.name,
                "input_price_per_1k": config.input_price_per_1k,
                "output_price_per_1k": config.output_price_per_1k,
                "max_tokens": config.max_tokens,
                "supports_embeddings": config.supports_embeddings,
            }
            for model_name, config in self.MODEL_CONFIGS.items()
        }

    def estimate_cost_for_text(
        self,
        input_text: str,
        model_name: str,
        estimated_output_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Estimate cost for processing given text.

        Args:
            input_text: Input text to process
            model_name: Model to use for processing
            estimated_output_length: Estimated output length in characters

        Returns:
            Cost estimation with breakdown
        """
        input_tokens = self.count_tokens(input_text, model_name)

        # Estimate output tokens
        output_tokens = 0
        if estimated_output_length:
            # Rough estimate: 1 token ≈ 4 characters
            output_tokens = estimated_output_length // 4

        token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        cost = self.calculate_cost(model_name, token_usage)

        config = self.MODEL_CONFIGS.get(model_name)

        return {
            "estimated_cost": cost,
            "token_usage": token_usage.model_dump(),
            "model_info": {
                "name": config.name if config else "Unknown",
                "max_tokens": config.max_tokens if config else 0,
            },
            "within_limits": input_tokens <= (config.max_tokens if config else 0),
        }

    def compare_model_costs(
        self, token_usage: TokenUsage, models: Optional[list[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare costs across different models for the same token usage.

        Args:
            token_usage: Token usage to compare
            models: List of models to compare (defaults to all models)

        Returns:
            Dictionary with cost comparison
        """
        if models is None:
            models = list(self.MODEL_CONFIGS.keys())

        comparison = {}

        for model_name in models:
            config = self.MODEL_CONFIGS.get(model_name)
            if not config:
                continue

            cost = self.calculate_cost(model_name, token_usage)

            comparison[model_name] = {
                "name": config.name,
                "cost": cost,
                "input_price_per_1k": config.input_price_per_1k,
                "output_price_per_1k": config.output_price_per_1k,
                "supports_embeddings": config.supports_embeddings,
            }

        # Sort by cost
        sorted_comparison = dict(sorted(comparison.items(), key=lambda x: x[1]["cost"]))

        return {
            "token_usage": token_usage.model_dump(),
            "models": sorted_comparison,
            "cheapest": (
                min(comparison.items(), key=lambda x: x[1]["cost"])[0]
                if comparison
                else None
            ),
            "most_expensive": (
                max(comparison.items(), key=lambda x: x[1]["cost"])[0]
                if comparison
                else None
            ),
        }

    def update_model_pricing(
        self,
        model_name: str,
        input_price_per_1k: Optional[float] = None,
        output_price_per_1k: Optional[float] = None,
    ) -> bool:
        """
        Update pricing for a specific model.

        Args:
            model_name: Model name to update
            input_price_per_1k: New input price per 1K tokens
            output_price_per_1k: New output price per 1K tokens

        Returns:
            True if updated successfully
        """
        if model_name not in self.MODEL_CONFIGS:
            logger.error(f"Model {model_name} not found")
            return False

        config = self.MODEL_CONFIGS[model_name]

        if input_price_per_1k is not None:
            config.input_price_per_1k = input_price_per_1k

        if output_price_per_1k is not None:
            config.output_price_per_1k = output_price_per_1k

        logger.info(f"Updated pricing for {model_name}")
        return True
