"""
Simple unit tests for core modules to verify basic functionality.
"""
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.unit
class TestCoreModules:
    """Basic tests for core module imports and initialization."""

    def test_text_processor_import(self):
        """Test that text_processor module can be imported."""
        from core import text_processor
        assert text_processor is not None

    def test_database_handler_import(self):
        """Test that database_handler module can be imported."""
        from core import database_handler
        assert database_handler is not None

    def test_quality_assessor_import(self):
        """Test that quality_assessor module can be imported."""
        from core import quality_assessor
        assert quality_assessor is not None

    def test_duplicate_detector_import(self):
        """Test that duplicate_detector module can be imported."""
        from core import duplicate_detector
        assert duplicate_detector is not None

    def test_processing_orchestrator_import(self):
        """Test that processing_orchestrator module can be imported."""
        from core import processing_orchestrator
        assert processing_orchestrator is not None

    @patch('core.text_processor.OpenAI')
    @patch('core.text_processor.tiktoken')
    def test_enhanced_text_processor_creation(self, mock_tiktoken, mock_openai):
        """Test creating an EnhancedTextProcessor instance."""
        from core.text_processor import EnhancedTextProcessor

        # Mock tiktoken encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_encoding

        processor = EnhancedTextProcessor(api_key="test-key")
        assert processor is not None
        assert processor.api_key == "test-key"

    def test_processing_config(self):
        """Test ProcessingConfig model."""
        from core.text_processor import ProcessingConfig

        config = ProcessingConfig(
            max_chunk_size=1000,
            chunk_overlap=100,
            extract_entities=True
        )

        assert config.max_chunk_size == 1000
        assert config.chunk_overlap == 100
        assert config.extract_entities is True

    def test_processing_quality(self):
        """Test ProcessingQuality model."""
        from core.text_processor import ProcessingQuality

        quality = ProcessingQuality(
            extraction_confidence=0.95,
            language_quality=0.90,
            completeness_score=0.88
        )

        assert quality.extraction_confidence == 0.95
        assert quality.overall_score > 0

    @patch('core.database_handler.supabase')
    def test_database_handler_creation(self, mock_supabase):
        """Test creating DatabaseHandler instance."""
        from core.database_handler import EnhancedDatabaseHandler

        # Mock Supabase client
        mock_client = MagicMock()
        mock_supabase.create_client.return_value = mock_client

        handler = EnhancedDatabaseHandler(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key"
        )

        assert handler is not None
        assert handler.client == mock_client

    def test_duplicate_detector_creation(self):
        """Test creating DuplicateDetector instance."""
        from core.duplicate_detector import DuplicateDetector

        detector = DuplicateDetector(threshold=0.85)
        assert detector is not None
        assert detector.threshold == 0.85

    def test_quality_assessor_creation(self):
        """Test creating QualityAssessor instance."""
        from core.quality_assessor import QualityAssessor

        assessor = QualityAssessor()
        assert assessor is not None

    @patch('core.processing_orchestrator.EnhancedTextProcessor')
    @patch('core.processing_orchestrator.EnhancedDatabaseHandler')
    def test_orchestrator_creation(self, mock_db, mock_processor):
        """Test creating ProcessingOrchestrator instance."""
        from core.processing_orchestrator import ProcessingOrchestrator

        orchestrator = ProcessingOrchestrator(
            text_processor=mock_processor,
            database_handler=mock_db
        )

        assert orchestrator is not None
        assert orchestrator.text_processor == mock_processor
        assert orchestrator.database_handler == mock_db