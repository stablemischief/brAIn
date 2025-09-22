"""
brAIn v2.0 Custom Validators
Custom validation logic for business rules and data integrity.
"""

import re
import hashlib
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

from pydantic import field_validator, ValidationInfo
from pydantic_core import core_schema
from pydantic_core.core_schema import ValidationInfo as CoreValidationInfo


# ========================================
# GOOGLE DRIVE VALIDATORS
# ========================================

class GoogleDriveIdValidator:
    """Validator for Google Drive file and folder IDs"""
    
    # Google Drive ID patterns
    DRIVE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{15,}$')
    LEGACY_ID_PATTERN = re.compile(r'^[0-9]{10,}$')
    
    @classmethod
    def validate(cls, value: str) -> str:
        """
        Validate Google Drive ID format.
        
        Args:
            value: The ID string to validate
            
        Returns:
            The validated ID
            
        Raises:
            ValueError: If ID format is invalid
        """
        if not isinstance(value, str):
            raise ValueError("Google Drive ID must be a string")
        
        value = value.strip()
        
        if not value:
            raise ValueError("Google Drive ID cannot be empty")
        
        # Check against known patterns
        if cls.DRIVE_ID_PATTERN.match(value) or cls.LEGACY_ID_PATTERN.match(value):
            return value
        
        # Additional checks for common formats
        if len(value) < 15:
            raise ValueError("Google Drive ID is too short")
        
        if len(value) > 100:
            raise ValueError("Google Drive ID is too long")
        
        # Check for invalid characters
        invalid_chars = set(value) - set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
        if invalid_chars:
            raise ValueError(f"Google Drive ID contains invalid characters: {', '.join(invalid_chars)}")
        
        return value


# ========================================
# CONTENT HASH VALIDATORS
# ========================================

class ContentHashValidator:
    """Validator for content hash values (SHA-256)"""
    
    SHA256_PATTERN = re.compile(r'^[a-fA-F0-9]{64}$')
    
    @classmethod
    def validate(cls, value: str) -> str:
        """
        Validate SHA-256 hash format.
        
        Args:
            value: The hash string to validate
            
        Returns:
            The validated hash in lowercase
            
        Raises:
            ValueError: If hash format is invalid
        """
        if not isinstance(value, str):
            raise ValueError("Content hash must be a string")
        
        value = value.strip().lower()
        
        if not value:
            raise ValueError("Content hash cannot be empty")
        
        if not cls.SHA256_PATTERN.match(value):
            raise ValueError("Content hash must be a valid 64-character SHA-256 hash")
        
        return value
    
    @classmethod
    def generate_hash(cls, content: Union[str, bytes]) -> str:
        """
        Generate SHA-256 hash from content.
        
        Args:
            content: Content to hash (string or bytes)
            
        Returns:
            SHA-256 hash in lowercase hex format
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        return hashlib.sha256(content).hexdigest()


# ========================================
# COST AND BUDGET VALIDATORS
# ========================================

class CostValidator:
    """Validator for cost values and budget calculations"""
    
    # Maximum reasonable cost values
    MAX_SINGLE_OPERATION_COST = Decimal('10.00')  # $10 per operation
    MAX_DAILY_COST = Decimal('1000.00')  # $1000 per day
    MAX_MONTHLY_BUDGET = Decimal('10000.00')  # $10,000 per month
    
    @classmethod
    def validate_cost(cls, value: Union[float, Decimal, str], field_name: str = "cost") -> Decimal:
        """
        Validate cost value.
        
        Args:
            value: Cost value to validate
            field_name: Name of the field for error messages
            
        Returns:
            Validated cost as Decimal
            
        Raises:
            ValueError: If cost is invalid
        """
        if value is None:
            return Decimal('0.00')
        
        try:
            cost = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError(f"{field_name} must be a valid number")
        
        if cost < 0:
            raise ValueError(f"{field_name} cannot be negative")
        
        # Check for reasonable upper bounds
        if cost > cls.MAX_SINGLE_OPERATION_COST:
            raise ValueError(f"{field_name} exceeds maximum allowed value of ${cls.MAX_SINGLE_OPERATION_COST}")
        
        # Ensure proper decimal places (4 decimal places for USD)
        quantized = cost.quantize(Decimal('0.0001'))
        
        return quantized
    
    @classmethod
    def validate_budget(cls, value: Union[float, Decimal, str]) -> Decimal:
        """
        Validate budget limit.
        
        Args:
            value: Budget value to validate
            
        Returns:
            Validated budget as Decimal
            
        Raises:
            ValueError: If budget is invalid
        """
        if value is None:
            return Decimal('100.00')  # Default budget
        
        try:
            budget = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("Budget must be a valid number")
        
        if budget < 0:
            raise ValueError("Budget cannot be negative")
        
        if budget > cls.MAX_MONTHLY_BUDGET:
            raise ValueError(f"Budget exceeds maximum allowed value of ${cls.MAX_MONTHLY_BUDGET}")
        
        return budget.quantize(Decimal('0.01'))


# ========================================
# TOKEN COUNT VALIDATORS
# ========================================

class TokenValidator:
    """Validator for token counts and token-related calculations"""
    
    # Reasonable token limits
    MAX_INPUT_TOKENS = 200000  # Large context models
    MAX_OUTPUT_TOKENS = 4000   # Typical completion length
    
    @classmethod
    def validate_token_count(cls, value: int, token_type: str = "token") -> int:
        """
        Validate token count.
        
        Args:
            value: Token count to validate
            token_type: Type of tokens (input/output) for error messages
            
        Returns:
            Validated token count
            
        Raises:
            ValueError: If token count is invalid
        """
        if not isinstance(value, int):
            raise ValueError(f"{token_type} count must be an integer")
        
        if value < 0:
            raise ValueError(f"{token_type} count cannot be negative")
        
        max_tokens = cls.MAX_INPUT_TOKENS if "input" in token_type.lower() else cls.MAX_OUTPUT_TOKENS
        
        if value > max_tokens:
            raise ValueError(f"{token_type} count exceeds maximum of {max_tokens:,}")
        
        return value


# ========================================
# EMBEDDING VALIDATORS
# ========================================

class EmbeddingValidator:
    """Validator for vector embeddings"""
    
    # Standard embedding dimensions
    OPENAI_SMALL_DIM = 1536
    OPENAI_LARGE_DIM = 3072
    SENTENCE_TRANSFORMER_DIM = 384
    
    SUPPORTED_DIMENSIONS = {OPENAI_SMALL_DIM, OPENAI_LARGE_DIM, SENTENCE_TRANSFORMER_DIM}
    
    @classmethod
    def validate_embedding(cls, value: List[float], expected_dim: int = OPENAI_SMALL_DIM) -> List[float]:
        """
        Validate vector embedding.
        
        Args:
            value: List of float values representing the embedding
            expected_dim: Expected number of dimensions
            
        Returns:
            Validated embedding vector
            
        Raises:
            ValueError: If embedding is invalid
        """
        if not isinstance(value, list):
            raise ValueError("Embedding must be a list of numbers")
        
        if len(value) != expected_dim:
            raise ValueError(f"Embedding must have exactly {expected_dim} dimensions, got {len(value)}")
        
        # Validate each dimension
        for i, val in enumerate(value):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Embedding value at index {i} must be a number, got {type(val).__name__}")
            
            # Check for invalid values
            if not (-10.0 <= val <= 10.0):  # Very generous range
                raise ValueError(f"Embedding value at index {i} is out of reasonable range: {val}")
            
            # Check for NaN or infinity
            if val != val:  # NaN check
                raise ValueError(f"Embedding value at index {i} is NaN")
            
            if abs(val) == float('inf'):
                raise ValueError(f"Embedding value at index {i} is infinite")
        
        return value


# ========================================
# FILE AND TEXT VALIDATORS
# ========================================

class FileValidator:
    """Validator for file-related data"""
    
    # Supported MIME types
    SUPPORTED_MIME_TYPES = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # docx
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',       # xlsx
        'application/vnd.openxmlformats-officedocument.presentationml.presentation', # pptx
        'text/plain',
        'text/markdown',
        'text/html',
        'text/csv',
        'application/json',
        'application/xml',
        'text/xml',
        'application/epub+zip',
        'application/rtf',
        'application/vnd.oasis.opendocument.text'  # odt
    }
    
    # File size limits (in bytes)
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
    MIN_FILE_SIZE = 1  # 1 byte
    
    @classmethod
    def validate_file_size(cls, size_bytes: int) -> int:
        """
        Validate file size.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Validated file size
            
        Raises:
            ValueError: If file size is invalid
        """
        if not isinstance(size_bytes, int):
            raise ValueError("File size must be an integer")
        
        if size_bytes < cls.MIN_FILE_SIZE:
            raise ValueError(f"File size must be at least {cls.MIN_FILE_SIZE} byte")
        
        if size_bytes > cls.MAX_FILE_SIZE:
            size_mb = size_bytes / (1024 * 1024)
            max_mb = cls.MAX_FILE_SIZE / (1024 * 1024)
            raise ValueError(f"File size of {size_mb:.1f}MB exceeds maximum of {max_mb}MB")
        
        return size_bytes
    
    @classmethod
    def validate_mime_type(cls, mime_type: str, strict: bool = False) -> str:
        """
        Validate MIME type.
        
        Args:
            mime_type: MIME type string
            strict: Whether to enforce supported types only
            
        Returns:
            Validated MIME type
            
        Raises:
            ValueError: If MIME type is invalid
        """
        if not isinstance(mime_type, str):
            raise ValueError("MIME type must be a string")
        
        mime_type = mime_type.strip().lower()
        
        if not mime_type:
            raise ValueError("MIME type cannot be empty")
        
        # Basic format validation
        if '/' not in mime_type:
            raise ValueError("MIME type must be in format 'type/subtype'")
        
        parts = mime_type.split('/')
        if len(parts) != 2:
            raise ValueError("MIME type must be in format 'type/subtype'")
        
        type_part, subtype_part = parts
        
        if not type_part or not subtype_part:
            raise ValueError("MIME type parts cannot be empty")
        
        # Check against supported types if strict
        if strict and mime_type not in cls.SUPPORTED_MIME_TYPES:
            raise ValueError(f"Unsupported MIME type: {mime_type}")
        
        return mime_type


# ========================================
# TEXT QUALITY VALIDATORS
# ========================================

class TextQualityValidator:
    """Validator for text quality metrics"""
    
    @classmethod
    def validate_quality_score(cls, score: float, field_name: str = "quality score") -> float:
        """
        Validate quality score (0.0 to 1.0).
        
        Args:
            score: Quality score to validate
            field_name: Name of the field for error messages
            
        Returns:
            Validated quality score
            
        Raises:
            ValueError: If score is invalid
        """
        if not isinstance(score, (int, float)):
            raise ValueError(f"{field_name} must be a number")
        
        if score < 0.0 or score > 1.0:
            raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        
        return float(score)
    
    @classmethod
    def assess_text_quality(cls, text: str) -> Dict[str, float]:
        """
        Assess text quality based on various metrics.
        
        Args:
            text: Text to assess
            
        Returns:
            Dictionary with quality metrics
        """
        if not text or not text.strip():
            return {
                'length_score': 0.0,
                'character_diversity': 0.0,
                'word_count': 0,
                'overall_quality': 0.0
            }
        
        text = text.strip()
        words = text.split()
        
        # Length score (prefer moderate length)
        length = len(text)
        if length < 50:
            length_score = length / 50.0  # Penalize very short text
        elif length <= 5000:
            length_score = 1.0  # Ideal range
        else:
            length_score = max(0.5, 1.0 - (length - 5000) / 50000)  # Penalize very long text
        
        # Character diversity
        unique_chars = set(text.lower())
        char_diversity = len(unique_chars) / max(100, len(text))  # Normalize by text length
        
        # Overall quality estimate
        overall_quality = (length_score * 0.4 + 
                          min(char_diversity, 1.0) * 0.3 + 
                          min(len(words) / max(length / 5, 1), 1.0) * 0.3)
        
        return {
            'length_score': round(length_score, 3),
            'character_diversity': round(char_diversity, 3),
            'word_count': len(words),
            'overall_quality': round(overall_quality, 3)
        }


# ========================================
# CONFIGURATION VALIDATORS
# ========================================

class ConfigurationValidator:
    """Validator for system configuration values"""
    
    @classmethod
    def validate_sync_frequency(cls, minutes: int) -> int:
        """
        Validate sync frequency in minutes.
        
        Args:
            minutes: Sync frequency in minutes
            
        Returns:
            Validated frequency
            
        Raises:
            ValueError: If frequency is invalid
        """
        if not isinstance(minutes, int):
            raise ValueError("Sync frequency must be an integer")
        
        if minutes < 5:
            raise ValueError("Sync frequency must be at least 5 minutes")
        
        if minutes > 1440:  # 24 hours
            raise ValueError("Sync frequency cannot exceed 24 hours (1440 minutes)")
        
        # Recommend reasonable intervals
        recommended = [5, 10, 15, 30, 60, 120, 240, 360, 720, 1440]
        if minutes not in recommended:
            closest = min(recommended, key=lambda x: abs(x - minutes))
            if abs(closest - minutes) > 10:  # Only warn if significantly different
                # This would ideally be a warning, but we'll allow it
                pass
        
        return minutes
    
    @classmethod
    def validate_batch_size(cls, size: int, operation: str = "processing") -> int:
        """
        Validate batch size for operations.
        
        Args:
            size: Batch size to validate
            operation: Type of operation for appropriate limits
            
        Returns:
            Validated batch size
            
        Raises:
            ValueError: If batch size is invalid
        """
        if not isinstance(size, int):
            raise ValueError("Batch size must be an integer")
        
        if size < 1:
            raise ValueError("Batch size must be at least 1")
        
        # Different limits for different operations
        max_sizes = {
            'processing': 50,
            'embedding': 100,
            'sync': 1000,
            'cleanup': 500
        }
        
        max_size = max_sizes.get(operation, 50)
        
        if size > max_size:
            raise ValueError(f"Batch size for {operation} cannot exceed {max_size}")
        
        return size


# ========================================
# PYDANTIC INTEGRATION HELPERS
# ========================================

def google_drive_id_validator(value: str) -> str:
    """Pydantic field validator for Google Drive IDs"""
    return GoogleDriveIdValidator.validate(value)


def content_hash_validator(value: Optional[str]) -> Optional[str]:
    """Pydantic field validator for content hashes"""
    if value is None:
        return None
    return ContentHashValidator.validate(value)


def cost_validator(value: Union[float, Decimal, str]) -> Decimal:
    """Pydantic field validator for cost values"""
    return CostValidator.validate_cost(value)


def embedding_validator(value: List[float], expected_dim: int = 1536) -> List[float]:
    """Pydantic field validator for embeddings"""
    return EmbeddingValidator.validate_embedding(value, expected_dim)


def quality_score_validator(value: float) -> float:
    """Pydantic field validator for quality scores"""
    return TextQualityValidator.validate_quality_score(value)


# ========================================
# BUSINESS RULE VALIDATORS
# ========================================

class BusinessRuleValidator:
    """Validators for business logic and rules"""
    
    @classmethod
    def validate_user_budget_consistency(cls, user_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate user budget consistency.
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}
        
        monthly_limit = user_data.get('monthly_budget_limit', 0)
        current_spend = user_data.get('current_month_spend', 0)
        
        if monthly_limit < current_spend:
            errors['budget_consistency'] = "Monthly budget limit cannot be less than current month spend"
        
        # Check for reasonable budget limits
        if monthly_limit > 10000:
            errors['budget_limit'] = "Monthly budget limit exceeds reasonable maximum of $10,000"
        
        return errors
    
    @classmethod
    def validate_document_processing_readiness(cls, document_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate if document is ready for processing.
        
        Args:
            document_data: Document data dictionary
            
        Returns:
            Dictionary of validation errors (empty if ready)
        """
        errors = {}
        
        # Check required fields
        required_fields = ['google_file_id', 'file_name', 'user_id', 'folder_id']
        for field in required_fields:
            if not document_data.get(field):
                errors[field] = f"{field} is required for processing"
        
        # Check file size
        file_size = document_data.get('file_size_bytes')
        if file_size and file_size > FileValidator.MAX_FILE_SIZE:
            errors['file_size'] = "File is too large for processing"
        
        # Check processing status
        status = document_data.get('processing_status')
        if status == 'processing':
            errors['processing_status'] = "Document is already being processed"
        
        return errors