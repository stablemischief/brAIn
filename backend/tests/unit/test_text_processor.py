"""
Unit tests for the TextProcessor module.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import tempfile

from core.text_processor import (
    TextProcessor,
    ProcessingResult,
    ProcessingConfig,
    FileType,
    ProcessingError,
)


@pytest.mark.unit
class TestTextProcessor:
    """Test suite for TextProcessor class."""

    @pytest.fixture
    def processor(self, mock_openai_client):
        """Create TextProcessor instance with mocked dependencies."""
        return TextProcessor(
            openai_client=mock_openai_client,
            config=ProcessingConfig(
                max_tokens=4000,
                embedding_model="text-embedding-ada-002",
                chat_model="gpt-4",
            ),
        )

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "This is a test document with some content for processing."

    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content."""
        return b"%PDF-1.4\nTest PDF content"

    def test_processor_initialization(self, processor):
        """Test TextProcessor initialization."""
        assert processor is not None
        assert processor.config.max_tokens == 4000
        assert processor.config.embedding_model == "text-embedding-ada-002"

    def test_detect_file_type_text(self, processor, temp_dir):
        """Test file type detection for text files."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("Sample text")

        file_type = processor.detect_file_type(str(text_file))
        assert file_type == FileType.TEXT

    def test_detect_file_type_pdf(self, processor, temp_dir):
        """Test file type detection for PDF files."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        file_type = processor.detect_file_type(str(pdf_file))
        assert file_type == FileType.PDF

    def test_detect_file_type_json(self, processor, temp_dir):
        """Test file type detection for JSON files."""
        json_file = temp_dir / "test.json"
        json_file.write_text('{"key": "value"}')

        file_type = processor.detect_file_type(str(json_file))
        assert file_type == FileType.JSON

    @pytest.mark.asyncio
    async def test_process_text_content(self, processor, sample_text):
        """Test processing text content."""
        result = await processor.process_text(sample_text)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.content == sample_text
        assert result.token_count > 0
        assert len(result.embedding) == 1536

    @pytest.mark.asyncio
    async def test_process_text_with_metadata_extraction(self, processor):
        """Test text processing with metadata extraction."""
        text = "Author: John Doe\nDate: 2024-01-01\n\nMain content here."

        result = await processor.process_text(text, extract_metadata=True)

        assert result.success is True
        assert "author" in result.metadata
        assert result.metadata["author"] == "John Doe"

    @pytest.mark.asyncio
    async def test_process_text_max_tokens_truncation(self, processor):
        """Test text truncation when exceeding max tokens."""
        long_text = "word " * 10000  # Very long text

        result = await processor.process_text(long_text)

        assert result.success is True
        assert result.token_count <= processor.config.max_tokens

    @pytest.mark.asyncio
    async def test_generate_embedding(self, processor, sample_text):
        """Test embedding generation."""
        embedding = await processor.generate_embedding(sample_text)

        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, processor):
        """Test embedding generation with empty text."""
        with pytest.raises(ProcessingError):
            await processor.generate_embedding("")

    def test_count_tokens(self, processor, sample_text):
        """Test token counting."""
        token_count = processor.count_tokens(sample_text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_empty(self, processor):
        """Test token counting with empty text."""
        token_count = processor.count_tokens("")
        assert token_count == 0

    @pytest.mark.asyncio
    async def test_extract_entities(self, processor):
        """Test entity extraction from text."""
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

        entities = await processor.extract_entities(text)

        assert isinstance(entities, list)
        assert len(entities) > 0
        assert any(e["type"] == "ORGANIZATION" for e in entities)
        assert any(e["type"] == "PERSON" for e in entities)
        assert any(e["type"] == "LOCATION" for e in entities)

    @pytest.mark.asyncio
    async def test_process_file_text(self, processor, temp_dir):
        """Test processing a text file."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("Test content")

        result = await processor.process_file(str(text_file))

        assert result.success is True
        assert result.content == "Test content"
        assert result.file_type == FileType.TEXT

    @pytest.mark.asyncio
    async def test_process_file_not_found(self, processor):
        """Test processing non-existent file."""
        with pytest.raises(ProcessingError):
            await processor.process_file("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_process_file_unsupported_type(self, processor, temp_dir):
        """Test processing unsupported file type."""
        binary_file = temp_dir / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        with pytest.raises(ProcessingError, match="Unsupported file type"):
            await processor.process_file(str(binary_file))

    def test_validate_config(self, processor):
        """Test configuration validation."""
        # Valid config should not raise
        processor.validate_config()

        # Invalid config should raise
        processor.config.max_tokens = -1
        with pytest.raises(ValueError):
            processor.validate_config()

    @pytest.mark.asyncio
    async def test_process_with_quality_assessment(self, processor, sample_text):
        """Test processing with quality assessment."""
        result = await processor.process_text(sample_text, assess_quality=True)

        assert result.success is True
        assert "quality_score" in result.metadata
        assert 0 <= result.metadata["quality_score"] <= 1

    @pytest.mark.asyncio
    async def test_batch_process_texts(self, processor):
        """Test batch processing of multiple texts."""
        texts = ["First document", "Second document", "Third document"]

        results = await processor.batch_process_texts(texts)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.content in texts for r in results)

    @pytest.mark.asyncio
    async def test_process_with_error_handling(self, processor, mock_openai_client):
        """Test error handling during processing."""
        # Simulate OpenAI API error
        mock_openai_client.embeddings.create.side_effect = Exception("API Error")

        result = await processor.process_text("Test text")

        assert result.success is False
        assert "error" in result.metadata
        assert "API Error" in result.metadata["error"]

    def test_clean_text(self, processor):
        """Test text cleaning functionality."""
        dirty_text = "  Test\n\n\ntext   with\textra\tspaces  "
        clean = processor.clean_text(dirty_text)

        assert clean == "Test text with extra spaces"

    def test_split_text_chunks(self, processor):
        """Test text splitting into chunks."""
        long_text = " ".join(["Word"] * 1000)

        chunks = processor.split_into_chunks(long_text, chunk_size=100)

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        assert all(processor.count_tokens(chunk) <= 100 for chunk in chunks)
