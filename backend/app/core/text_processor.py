"""
Enhanced Text Processing Module for brAIn v2.0 RAG Pipeline

This module provides enhanced text extraction, processing, and validation capabilities
based on the proven RAG Pipeline with modern AI validation, cost tracking, and
quality assessment features.

Author: BMad Team
"""

import os
import io
import csv
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from decimal import Decimal
import hashlib
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime

# Pydantic imports for validation
from pydantic import BaseModel, Field, validator, ValidationError
from pydantic import field_validator

# Core dependencies
try:
    import pypdf

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Additional format support
try:
    from pptx import Presentation

    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

try:
    from docx import Document

    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False

try:
    import xlrd

    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False

try:
    import olefile

    OLEFILE_AVAILABLE = True
except ImportError:
    OLEFILE_AVAILABLE = False

# Markdown conversion for hyperlink preservation
try:
    import mammoth
    import markdownify
    import pdfplumber

    MARKDOWN_CONVERSION_AVAILABLE = True
except ImportError:
    MARKDOWN_CONVERSION_AVAILABLE = False

# AI validation and cost tracking
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"

if DOTENV_AVAILABLE:
    load_dotenv(dotenv_path, override=True)

# =============================================================================
# PYDANTIC MODELS FOR AI VALIDATION
# =============================================================================


class ProcessingQuality(BaseModel):
    """Quality assessment for text processing operations."""

    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence in extraction quality"
    )
    text_length: int = Field(ge=0, description="Length of extracted text")
    estimated_accuracy: float = Field(
        ge=0.0, le=1.0, description="Estimated accuracy of extraction"
    )
    extraction_method: str = Field(description="Method used for extraction")
    format_specific_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Format-specific quality metrics"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings during processing"
    )
    errors: List[str] = Field(
        default_factory=list, description="Errors during processing"
    )

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v


class TextChunk(BaseModel):
    """Validated text chunk with metadata."""

    content: str = Field(min_length=1, description="Chunk content")
    chunk_index: int = Field(ge=0, description="Zero-based chunk index")
    chunk_size: int = Field(gt=0, description="Size of chunk in characters")
    overlap_size: int = Field(ge=0, description="Overlap with previous chunk")
    quality_score: float = Field(
        ge=0.0, le=1.0, description="Quality assessment for this chunk"
    )
    content_hash: str = Field(description="SHA-256 hash of content")
    language_detected: Optional[str] = Field(
        default=None, description="Detected language code"
    )

    def __init__(self, **data):
        if "content_hash" not in data and "content" in data:
            data["content_hash"] = hashlib.sha256(data["content"].encode()).hexdigest()
        if "chunk_size" not in data and "content" in data:
            data["chunk_size"] = len(data["content"])
        super().__init__(**data)


class FileProcessingResult(BaseModel):
    """Complete result of file processing with validation."""

    file_id: str = Field(description="Unique identifier for the file")
    file_name: str = Field(description="Original filename")
    mime_type: str = Field(description="MIME type of the file")
    file_size: int = Field(ge=0, description="File size in bytes")
    extracted_text: str = Field(description="Complete extracted text")
    text_chunks: List[TextChunk] = Field(description="Validated text chunks")
    processing_quality: ProcessingQuality = Field(description="Quality assessment")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    token_count: int = Field(ge=0, description="Estimated token count")
    cost_estimate: Decimal = Field(ge=0, description="Estimated processing cost")
    content_language: Optional[str] = Field(
        default=None, description="Detected content language"
    )
    content_hash: str = Field(description="SHA-256 hash of original content")
    extracted_entities: List[str] = Field(
        default_factory=list, description="Extracted entities"
    )
    detected_links: List[str] = Field(
        default_factory=list, description="Detected hyperlinks"
    )
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("cost_estimate", mode="before")
    @classmethod
    def validate_cost(cls, v):
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v


class ProcessingConfig(BaseModel):
    """Configuration for text processing operations."""

    chunk_size: int = Field(
        default=400, ge=50, le=8000, description="Default chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=0, ge=0, le=200, description="Overlap between chunks"
    )
    quality_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum quality threshold"
    )
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    max_file_size_mb: int = Field(
        default=100, ge=1, le=1000, description="Maximum file size in MB"
    )
    supported_formats: List[str] = Field(
        default_factory=lambda: [
            "pdf",
            "docx",
            "doc",
            "xlsx",
            "xls",
            "pptx",
            "txt",
            "html",
            "csv",
        ]
    )
    enable_language_detection: bool = Field(
        default=True, description="Enable language detection"
    )
    enable_entity_extraction: bool = Field(
        default=True, description="Enable entity extraction"
    )
    enable_duplicate_detection: bool = Field(
        default=True, description="Enable duplicate detection"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        if hasattr(info.data, "chunk_size") and v >= info.data["chunk_size"]:
            raise ValueError("Overlap must be less than chunk size")
        return v


# =============================================================================
# ENHANCED TEXT PROCESSING FUNCTIONS
# =============================================================================


class EnhancedTextProcessor:
    """Enhanced text processor with AI validation and quality assessment."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize with configuration."""
        self.config = config or ProcessingConfig()
        self.openai_client = None
        self._token_encoder = None

        # Initialize token encoder for cost calculations
        if TIKTOKEN_AVAILABLE:
            try:
                self._token_encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(f"Warning: Could not initialize tiktoken encoder: {e}")

    def get_openai_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self.openai_client is None:
            api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                if DOTENV_AVAILABLE:
                    load_dotenv(dotenv_path, override=True)
                api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")

            if api_key:
                base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
                self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                raise ValueError(
                    "EMBEDDING_API_KEY or OPENAI_API_KEY not found in environment variables"
                )
        return self.openai_client

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        if self._token_encoder:
            try:
                return len(self._token_encoder.encode(text))
            except Exception:
                pass

        # Fallback estimation: ~4 characters per token
        return len(text) // 4

    def estimate_processing_cost(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> Decimal:
        """Estimate cost for processing text."""
        if not self.config.enable_cost_tracking:
            return Decimal("0.00")

        token_count = self.estimate_token_count(text)

        # Cost per 1K tokens for different models (as of 2024)
        cost_per_1k_tokens = {
            "text-embedding-3-small": Decimal("0.00002"),
            "text-embedding-3-large": Decimal("0.00013"),
            "text-embedding-ada-002": Decimal("0.00010"),
        }

        rate = cost_per_1k_tokens.get(
            model, cost_per_1k_tokens["text-embedding-3-small"]
        )
        return (Decimal(token_count) / Decimal("1000")) * rate

    def sanitize_text(self, text: str) -> str:
        """
        Enhanced text sanitization with quality tracking.

        Args:
            text: The text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        original_length = len(text)

        # Remove null characters
        text = text.replace("\x00", "")

        # Remove other control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        # Normalize whitespace (multiple spaces to single space)
        text = re.sub(r" +", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Log significant text loss
        if len(text) < original_length * 0.8:
            print(
                f"Warning: Significant text loss during sanitization ({original_length} -> {len(text)})"
            )

        return text

    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of text content."""
        if not self.config.enable_language_detection or len(text) < 50:
            return None

        try:
            # Simple language detection based on common patterns
            # In production, consider using langdetect or similar libraries

            # Check for common English words
            english_words = [
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
            ]
            english_count = sum(
                1 for word in english_words if f" {word} " in text.lower()
            )

            if english_count > 3:
                return "en"

            return "unknown"
        except Exception:
            return None

    def extract_entities(self, text: str) -> List[str]:
        """Extract basic entities from text."""
        if not self.config.enable_entity_extraction or len(text) < 20:
            return []

        entities = []

        try:
            # Extract email addresses
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            emails = re.findall(email_pattern, text)
            entities.extend([f"EMAIL:{email}" for email in emails])

            # Extract URLs
            url_pattern = r'https?://[^\s<>"\'`]+|www\.[^\s<>"\'`]+'
            urls = re.findall(url_pattern, text)
            entities.extend([f"URL:{url}" for url in urls])

            # Extract potential dates (basic patterns)
            date_pattern = (
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
            )
            dates = re.findall(date_pattern, text)
            entities.extend([f"DATE:{date}" for date in dates])

            return entities[:50]  # Limit to 50 entities
        except Exception:
            return []

    def detect_hyperlinks(self, text: str) -> List[str]:
        """Detect hyperlinks in text."""
        links = []

        try:
            # Extract URLs
            url_pattern = r'https?://[^\s<>"\'`]+|www\.[^\s<>"\'`]+'
            urls = re.findall(url_pattern, text)
            links.extend(urls)

            # Extract markdown-style links
            markdown_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
            markdown_links = re.findall(markdown_pattern, text)
            links.extend([url for _, url in markdown_links])

            return list(set(links))  # Remove duplicates
        except Exception:
            return []

    def assess_extraction_quality(
        self,
        original_size: int,
        extracted_text: str,
        extraction_method: str,
        warnings: List[str],
        errors: List[str],
    ) -> ProcessingQuality:
        """Assess the quality of text extraction."""

        text_length = len(extracted_text)

        # Base confidence calculation
        if text_length == 0:
            confidence_score = 0.0
        elif original_size == 0:
            confidence_score = 0.5  # Unknown original size
        else:
            # Higher confidence for reasonable text extraction ratios
            ratio = text_length / original_size
            if 0.01 <= ratio <= 0.5:  # Reasonable compression
                confidence_score = 0.8
            elif ratio > 0.5:  # Very high extraction
                confidence_score = 0.9
            else:  # Very low extraction
                confidence_score = 0.3

        # Reduce confidence for errors and warnings
        confidence_score -= len(errors) * 0.2
        confidence_score -= len(warnings) * 0.1
        confidence_score = max(0.0, min(1.0, confidence_score))

        # Estimate accuracy based on method and results
        accuracy_scores = {
            "docx_xml": 0.95,
            "pdf_pdfplumber": 0.90,
            "pdf_pypdf": 0.85,
            "xlsx_openpyxl": 0.95,
            "pptx_native": 0.90,
            "csv_native": 0.98,
            "html_markdownify": 0.85,
            "txt_direct": 0.99,
            "fallback": 0.60,
        }

        estimated_accuracy = accuracy_scores.get(extraction_method, 0.70)

        return ProcessingQuality(
            confidence_score=confidence_score,
            text_length=text_length,
            estimated_accuracy=estimated_accuracy,
            extraction_method=extraction_method,
            warnings=warnings,
            errors=errors,
            format_specific_metrics={
                "original_size": original_size,
                "compression_ratio": text_length / max(original_size, 1),
                "has_structured_content": any(
                    marker in extracted_text.lower()
                    for marker in ["table", "sheet", "column", "row"]
                ),
            },
        )

    def create_validated_chunks(
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[TextChunk]:
        """Create validated text chunks with quality assessment."""
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap

        # Sanitize the text first
        text = self.sanitize_text(text)

        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_content = text[start:end]

            # Assess chunk quality
            quality_score = self._assess_chunk_quality(chunk_content)

            # Only include chunks that meet quality threshold
            if quality_score >= self.config.quality_threshold:
                chunk = TextChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    overlap_size=min(overlap, start) if start > 0 else 0,
                    quality_score=quality_score,
                    language_detected=self.detect_language(chunk_content),
                )
                chunks.append(chunk)
                chunk_index += 1

            start = end - overlap if overlap > 0 else end

        return chunks

    def _assess_chunk_quality(self, chunk: str) -> float:
        """Assess the quality of a text chunk."""
        if not chunk or len(chunk) < 10:
            return 0.0

        quality_score = 0.8  # Base score

        # Check for meaningful content
        word_count = len(chunk.split())
        if word_count < 3:
            quality_score -= 0.3
        elif word_count > 10:
            quality_score += 0.1

        # Check for excessive repetition
        words = chunk.lower().split()
        if len(set(words)) < len(words) * 0.5:  # High repetition
            quality_score -= 0.2

        # Check for reasonable character distribution
        alpha_ratio = sum(1 for c in chunk if c.isalpha()) / len(chunk)
        if alpha_ratio < 0.3:  # Too few letters
            quality_score -= 0.2

        return max(0.0, min(1.0, quality_score))

    def extract_text_from_docx(
        self, file_content: bytes
    ) -> Tuple[str, List[str], List[str]]:
        """Enhanced DOCX text extraction with error tracking."""
        warnings = []
        errors = []

        try:
            # DOCX files are ZIP archives
            with zipfile.ZipFile(io.BytesIO(file_content)) as zipf:
                # The main document content is in word/document.xml
                if "word/document.xml" in zipf.namelist():
                    with zipf.open("word/document.xml") as xml_file:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()

                        # Extract text from all w:t elements
                        texts = []
                        for elem in root.iter(
                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                        ):
                            if elem.text:
                                texts.append(elem.text)

                        extracted_text = " ".join(texts)
                        if len(extracted_text) < 50 and len(file_content) > 1000:
                            warnings.append(
                                "Very little text extracted from large DOCX file"
                            )

                        return self.sanitize_text(extracted_text), warnings, errors
                else:
                    errors.append("word/document.xml not found in DOCX file")
                    return "", warnings, errors
        except Exception as e:
            errors.append(f"DOCX extraction error: {str(e)}")
            # Fallback to basic text extraction
            try:
                fallback_text = file_content.decode("utf-8", errors="ignore")
                warnings.append("Used fallback text extraction for DOCX")
                return self.sanitize_text(fallback_text), warnings, errors
            except Exception:
                errors.append("Fallback extraction also failed")
                return "", warnings, errors

    def extract_text_from_pdf(
        self, file_content: bytes
    ) -> Tuple[str, List[str], List[str]]:
        """Enhanced PDF text extraction with quality assessment."""
        warnings = []
        errors = []

        if not PYPDF_AVAILABLE:
            errors.append("pypdf not available, cannot extract from PDF")
            return "", warnings, errors

        try:
            # Create a temporary file to save the PDF content
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(file_content)
                temp_pdf_path = temp_pdf.name

            # Try pdfplumber first (better quality)
            extracted_text = ""
            method_used = "fallback"

            if MARKDOWN_CONVERSION_AVAILABLE:
                try:
                    with pdfplumber.open(temp_pdf_path) as pdf:
                        text_parts = []
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                        extracted_text = " ".join(text_parts)
                        method_used = "pdfplumber"
                except Exception as e:
                    warnings.append(f"pdfplumber failed: {str(e)}, trying pypdf")

            # Fallback to pypdf if pdfplumber failed or unavailable
            if not extracted_text:
                try:
                    with open(temp_pdf_path, "rb") as pdf_file:
                        pdf_reader = pypdf.PdfReader(pdf_file)
                        text_parts = []

                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)

                        extracted_text = " ".join(text_parts)
                        method_used = "pypdf"
                except Exception as e:
                    errors.append(f"pypdf extraction failed: {str(e)}")

            # Clean up the temporary file
            os.remove(temp_pdf_path)

            # Quality assessment
            if len(extracted_text) < 100 and len(file_content) > 10000:
                warnings.append(
                    f"Very little text extracted from PDF using {method_used}"
                )

            return self.sanitize_text(extracted_text), warnings, errors

        except Exception as e:
            errors.append(f"PDF processing error: {str(e)}")
            return "", warnings, errors

    def extract_text_from_file(
        self, file_content: bytes, mime_type: str, file_name: str
    ) -> Tuple[str, str, List[str], List[str]]:
        """
        Enhanced file text extraction with method tracking.

        Args:
            file_content: Binary content of the file
            mime_type: MIME type of the file
            file_name: Name of the file

        Returns:
            Tuple of (extracted_text, extraction_method, warnings, errors)
        """
        warnings = []
        errors = []
        text = ""
        method = "unknown"

        try:
            # Route to appropriate extraction method
            if (
                mime_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                or file_name.endswith(".docx")
            ):
                text, method_warnings, method_errors = self.extract_text_from_docx(
                    file_content
                )
                method = "docx_xml"
                warnings.extend(method_warnings)
                errors.extend(method_errors)

            elif mime_type == "application/pdf" or file_name.endswith(".pdf"):
                text, method_warnings, method_errors = self.extract_text_from_pdf(
                    file_content
                )
                method = "pdf_extraction"
                warnings.extend(method_warnings)
                errors.extend(method_errors)

            elif mime_type in [
                "text/plain",
                "text/html",
                "text/csv",
            ] or file_name.endswith((".txt", ".html", ".csv")):
                try:
                    text = file_content.decode("utf-8", errors="replace")
                    method = "txt_direct"
                except Exception as e:
                    errors.append(f"Text decoding error: {str(e)}")
                    text = ""
                    method = "fallback"

            # Add more extraction methods here for other formats...
            else:
                # Fallback to basic text extraction
                try:
                    text = file_content.decode("utf-8", errors="replace")
                    method = "fallback"
                    warnings.append(f"Using fallback extraction for {mime_type}")
                except Exception as e:
                    errors.append(f"Fallback extraction failed: {str(e)}")
                    text = ""
                    method = "failed"

            return self.sanitize_text(text), method, warnings, errors

        except Exception as e:
            errors.append(f"File extraction error: {str(e)}")
            return "", "error", warnings, errors

    def process_file_with_validation(
        self,
        file_content: bytes,
        file_name: str,
        file_id: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> FileProcessingResult:
        """
        Process a file with complete AI validation and quality assessment.

        Args:
            file_content: Binary content of the file
            file_name: Name of the file
            file_id: Optional unique identifier
            mime_type: Optional MIME type

        Returns:
            Complete processing result with validation
        """
        start_time = datetime.utcnow()

        # Generate file ID if not provided
        if not file_id:
            file_id = hashlib.sha256(file_content + file_name.encode()).hexdigest()[:16]

        # Detect MIME type if not provided
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_name)
            mime_type = mime_type or "application/octet-stream"

        # Validate file size
        file_size = len(file_content)
        max_size = self.config.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            raise ValidationError(
                f"File size {file_size} exceeds maximum {max_size} bytes"
            )

        # Extract text with quality tracking
        extracted_text, extraction_method, warnings, errors = (
            self.extract_text_from_file(file_content, mime_type, file_name)
        )

        # Create validated chunks
        text_chunks = self.create_validated_chunks(extracted_text)

        # Assess processing quality
        processing_quality = self.assess_extraction_quality(
            file_size, extracted_text, extraction_method, warnings, errors
        )

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Estimate costs
        token_count = self.estimate_token_count(extracted_text)
        cost_estimate = self.estimate_processing_cost(extracted_text)

        # Detect content features
        content_language = self.detect_language(extracted_text)
        extracted_entities = self.extract_entities(extracted_text)
        detected_links = self.detect_hyperlinks(extracted_text)

        # Generate content hash
        content_hash = hashlib.sha256(file_content).hexdigest()

        # Create and validate result
        try:
            result = FileProcessingResult(
                file_id=file_id,
                file_name=file_name,
                mime_type=mime_type,
                file_size=file_size,
                extracted_text=extracted_text,
                text_chunks=text_chunks,
                processing_quality=processing_quality,
                processing_time=processing_time,
                token_count=token_count,
                cost_estimate=cost_estimate,
                content_language=content_language,
                content_hash=content_hash,
                extracted_entities=extracted_entities,
                detected_links=detected_links,
            )
            return result

        except ValidationError as e:
            print(f"Validation error in processing result: {e}")
            raise

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for text chunks with cost tracking.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Sanitize all texts before embedding
        sanitized_texts = [self.sanitize_text(text) for text in texts]
        # Filter out empty strings
        sanitized_texts = [text for text in sanitized_texts if text]

        if not sanitized_texts:
            return []

        client = self.get_openai_client()
        model = os.getenv("EMBEDDING_MODEL_CHOICE", "text-embedding-3-small")

        response = client.embeddings.create(model=model, input=sanitized_texts)

        # Extract the embedding vectors from the response
        embeddings = [item.embedding for item in response.data]

        return embeddings


# =============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================

# Global processor instance for backward compatibility
_default_processor = None


def get_default_processor() -> EnhancedTextProcessor:
    """Get the default processor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = EnhancedTextProcessor()
    return _default_processor


def sanitize_text(text: str) -> str:
    """Sanitize text using default processor."""
    return get_default_processor().sanitize_text(text)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 0) -> List[str]:
    """Create text chunks using default processor (legacy compatibility)."""
    processor = get_default_processor()
    validated_chunks = processor.create_validated_chunks(text, chunk_size, overlap)
    return [chunk.content for chunk in validated_chunks]


def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings using default processor."""
    return get_default_processor().create_embeddings(texts)


def extract_text_from_file(file_content: bytes, mime_type: str, file_name: str) -> str:
    """Extract text from file using default processor (legacy compatibility)."""
    processor = get_default_processor()
    text, _, _, _ = processor.extract_text_from_file(file_content, mime_type, file_name)
    return text


def is_tabular_file(file_path: str) -> bool:
    """Check if file contains tabular data (CSV, TSV, Excel)."""
    import mimetypes

    mime_type, _ = mimetypes.guess_type(file_path)

    tabular_types = [
        "text/csv",
        "text/tab-separated-values",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]

    if mime_type in tabular_types:
        return True

    # Check by extension if MIME type not found
    extensions = [".csv", ".tsv", ".xls", ".xlsx"]
    return any(file_path.lower().endswith(ext) for ext in extensions)


def extract_schema_from_csv(content: str) -> Dict[str, str]:
    """Extract schema from CSV content."""
    import csv
    from io import StringIO

    reader = csv.DictReader(StringIO(content))
    headers = reader.fieldnames or []

    # Simple type inference (can be enhanced)
    schema = {}
    for header in headers:
        schema[header] = "string"  # Default to string type

    return schema


def extract_rows_from_csv(content: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Extract sample rows from CSV content."""
    import csv
    from io import StringIO

    reader = csv.DictReader(StringIO(content))
    rows = []

    for i, row in enumerate(reader):
        if i >= limit:
            break
        rows.append(dict(row))

    return rows
