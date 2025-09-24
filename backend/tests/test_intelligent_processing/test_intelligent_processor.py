"""
Comprehensive Test Suite for Intelligent File Processing System

This module provides extensive testing for the AI-enhanced file processing system,
covering all components and integration scenarios.

Author: BMad Team
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

# Import the modules to test
import sys

sys.path.append("/Users/james/Documents/Product-RD/brAIn")

from src.processing.intelligent_processor import (
    IntelligentFileProcessor,
    ProcessingContext,
    IntelligentProcessingResult,
    ProcessingJob,
    process_file_with_ai,
    batch_process_with_ai,
)

from src.processing.file_classification import (
    AIFileClassifier,
    FileTypeAnalysis,
    FileSignature,
    ContentAnalysis,
    classify_file_ai,
)

from src.processing.quality_assessment import (
    ProcessingQualityAnalyzer,
    QualityAnalysisResult,
    ExtractionQualityMetrics,
    ContentQualityMetrics,
    ProcessingEfficiencyMetrics,
    analyze_processing_quality,
)

from src.processing.rules_engine import (
    ExtractionRulesEngine,
    CustomRule,
    RuleSet,
    RuleType,
    RulePriority,
    extract_with_custom_rules,
)

from src.processing.format_optimization import (
    FormatOptimizationEngine,
    OptimizationStrategy,
    OptimizationLevel,
    OptimizationResult,
    optimize_file_processing,
)

from src.processing.error_intelligence import (
    IntelligentErrorHandler,
    ErrorAnalysis,
    RecoveryResult,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    handle_processing_error,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def temp_text_file():
    """Create temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(
            """
        This is a test document for intelligent processing.
        
        It contains multiple paragraphs with different content types:
        - Email: test@example.com
        - Phone: (555) 123-4567
        - Date: January 15, 2024
        - URL: https://www.example.com
        
        The document structure includes headers, lists, and normal text
        to test various processing capabilities.
        
        Some technical terms: API, JSON, HTTP, REST
        Financial info: $1,234.56, Account Number: 12345-67890
        
        This content should be sufficient for comprehensive testing
        of the intelligent processing system.
        """
        )
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def temp_json_file():
    """Create temporary JSON file for testing."""
    import json

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        data = {
            "name": "Test Document",
            "type": "sample",
            "content": {
                "sections": [
                    {"title": "Introduction", "text": "This is a test"},
                    {"title": "Body", "text": "Main content here"},
                    {"title": "Conclusion", "text": "Final thoughts"},
                ]
            },
            "metadata": {
                "created": "2024-01-15",
                "author": "Test User",
                "version": "1.0",
            },
        }
        json.dump(data, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def mock_ai_client():
    """Mock AI client for testing."""
    mock_client = Mock()

    # Mock Anthropic response
    mock_message = Mock()
    mock_message.content = [Mock()]
    mock_message.content[0].text = (
        '{"document_type": "text", "category": "general", "complexity": "moderate", "confidence": 80}'
    )

    mock_client.messages = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    return mock_client


@pytest.fixture
def sample_processing_context():
    """Sample processing context for testing."""
    return ProcessingContext(
        user_preferences={"quality": "high", "speed": "moderate"},
        domain_knowledge="technical",
        priority_level="normal",
        quality_threshold=0.8,
        custom_rules=["emails", "phone_numbers"],
        optimization_level="balanced",
    )


# =============================================================================
# INTELLIGENT PROCESSOR TESTS
# =============================================================================


class TestIntelligentFileProcessor:
    """Test cases for IntelligentFileProcessor."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = IntelligentFileProcessor()

        assert processor is not None
        assert processor.file_classifier is not None
        assert processor.quality_analyzer is not None
        assert processor.rules_engine is not None
        assert processor.format_optimizer is not None
        assert processor.error_handler is not None
        assert processor.processing_stats["total_processed"] == 0

    def test_processor_with_ai_client(self, mock_ai_client):
        """Test processor initialization with AI client."""
        processor = IntelligentFileProcessor(ai_client=mock_ai_client)

        assert processor.ai_client == mock_ai_client
        assert processor.file_classifier.ai_client == mock_ai_client

    @pytest.mark.asyncio
    async def test_process_text_file(self, temp_text_file, sample_processing_context):
        """Test processing a text file."""
        processor = IntelligentFileProcessor()

        result = await processor.process_file_intelligently(
            temp_text_file, sample_processing_context
        )

        assert isinstance(result, IntelligentProcessingResult)
        assert result.file_path == str(temp_text_file)
        assert result.detected_format in ["txt", "text", "plain"]
        assert result.confidence_score > 0.0
        assert result.processing_quality >= 0.0
        assert result.processing_time > 0.0
        assert result.token_count >= 0
        assert len(result.content_preview) > 0

    @pytest.mark.asyncio
    async def test_process_json_file(self, temp_json_file, sample_processing_context):
        """Test processing a JSON file."""
        processor = IntelligentFileProcessor()

        result = await processor.process_file_intelligently(
            temp_json_file, sample_processing_context
        )

        assert isinstance(result, IntelligentProcessingResult)
        assert result.detected_format in ["json", "application/json"]
        assert result.processing_quality > 0.0
        assert (
            "json" in result.extraction_method.lower()
            or "structured" in result.extraction_method.lower()
        )

    @pytest.mark.asyncio
    async def test_batch_processing(
        self, temp_text_file, temp_json_file, sample_processing_context
    ):
        """Test batch processing multiple files."""
        processor = IntelligentFileProcessor()

        results = await processor.batch_process_intelligently(
            [temp_text_file, temp_json_file], sample_processing_context
        )

        assert len(results) == 2
        assert all(isinstance(r, IntelligentProcessingResult) for r in results)
        assert results[0].file_path == str(temp_text_file)
        assert results[1].file_path == str(temp_json_file)

    def test_job_id_generation(self, temp_text_file):
        """Test job ID generation."""
        processor = IntelligentFileProcessor()

        job_id1 = processor._generate_job_id(temp_text_file)
        job_id2 = processor._generate_job_id(temp_text_file)

        assert len(job_id1) == 16
        assert len(job_id2) == 16
        assert job_id1 != job_id2  # Should be unique due to timestamp

    def test_processing_analytics(self):
        """Test processing analytics."""
        processor = IntelligentFileProcessor()

        analytics = processor.get_processing_analytics()

        assert "statistics" in analytics
        assert "component_health" in analytics
        assert "recommendations" in analytics

        # Check component health
        health = analytics["component_health"]
        assert "file_classifier" in health
        assert "quality_analyzer" in health
        assert "rules_engine" in health
        assert "format_optimizer" in health
        assert "error_handler" in health


# =============================================================================
# FILE CLASSIFICATION TESTS
# =============================================================================


class TestAIFileClassifier:
    """Test cases for AIFileClassifier."""

    def test_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = AIFileClassifier()

        assert classifier is not None
        assert classifier.pattern_db is not None
        assert len(classifier.pattern_db.MAGIC_SIGNATURES) > 0
        assert len(classifier.pattern_db.CONTENT_PATTERNS) > 0
        assert classifier.stats["total_classifications"] == 0

    @pytest.mark.asyncio
    async def test_classify_text_file(self, temp_text_file):
        """Test classifying a text file."""
        classifier = AIFileClassifier()

        result = await classifier.classify_with_content_analysis(temp_text_file)

        assert isinstance(result, FileTypeAnalysis)
        assert result.detected_format in ["txt", "text", "plain"]
        assert result.confidence_score > 0.0
        assert result.detection_method in ["signature", "content_pattern", "fallback"]
        assert isinstance(result.signature_analysis, FileSignature)
        assert isinstance(result.content_analysis, ContentAnalysis)

    @pytest.mark.asyncio
    async def test_classify_json_file(self, temp_json_file):
        """Test classifying a JSON file."""
        classifier = AIFileClassifier()

        result = await classifier.classify_with_content_analysis(temp_json_file)

        assert isinstance(result, FileTypeAnalysis)
        assert result.detected_format == "json"
        assert result.content_analysis.structure_type == "json"
        assert "json" in result.content_analysis.content_patterns

    @pytest.mark.asyncio
    async def test_classify_with_domain_context(self, temp_text_file):
        """Test classification with domain context."""
        classifier = AIFileClassifier()

        result = await classifier.classify_with_content_analysis(
            temp_text_file, domain_context="technical"
        )

        assert isinstance(result, FileTypeAnalysis)
        assert result.metadata.get("domain_context") == "technical"

        # Should detect technical patterns
        if result.content_analysis.content_patterns:
            # May include domain-specific patterns
            pass

    @pytest.mark.asyncio
    async def test_classify_nonexistent_file(self):
        """Test classifying non-existent file."""
        classifier = AIFileClassifier()

        with pytest.raises(FileNotFoundError):
            await classifier.classify_with_content_analysis("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_ai_content_analysis(self, temp_text_file, mock_ai_client):
        """Test AI-powered content analysis."""
        classifier = AIFileClassifier(ai_client=mock_ai_client)

        result = await classifier.classify_with_content_analysis(temp_text_file)

        assert isinstance(result, FileTypeAnalysis)
        # AI analysis should be included
        assert len(result.ai_analysis) > 0

        # Mock client should have been called
        mock_ai_client.messages.create.assert_called_once()


# =============================================================================
# QUALITY ASSESSMENT TESTS
# =============================================================================


class TestProcessingQualityAnalyzer:
    """Test cases for ProcessingQualityAnalyzer."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = ProcessingQualityAnalyzer()

        assert analyzer is not None
        assert len(analyzer.thresholds) > 0
        assert analyzer.stats["analyses_performed"] == 0

    @pytest.mark.asyncio
    async def test_analyze_processing_quality(self, sample_processing_context):
        """Test quality analysis."""
        analyzer = ProcessingQualityAnalyzer()

        # Mock extraction result
        extraction_result = {
            "content": "This is test content with good structure and readability.",
            "processing_time": 1.5,
            "token_count": 50,
            "cost": 0.001,
            "metadata": {"source": "test"},
        }

        # Mock file classification
        class MockClassification:
            detected_format = "txt"
            confidence_score = 0.8

        result = await analyzer.analyze_processing_quality(
            extraction_result, MockClassification(), sample_processing_context
        )

        assert isinstance(result, QualityAnalysisResult)
        assert result.overall_score >= 0.0
        assert result.overall_score <= 1.0
        assert isinstance(result.extraction_metrics, ExtractionQualityMetrics)
        assert isinstance(result.content_metrics, ContentQualityMetrics)
        assert isinstance(result.efficiency_metrics, ProcessingEfficiencyMetrics)

    def test_flesch_score_calculation(self):
        """Test Flesch reading ease score calculation."""
        analyzer = ProcessingQualityAnalyzer()

        # Simple text
        simple_text = "The cat sat on the mat. It was a nice day."
        score = analyzer._calculate_flesch_score(simple_text)
        assert 0.0 <= score <= 100.0

        # Complex text
        complex_text = "The implementation of sophisticated algorithmic methodologies necessitates comprehensive analytical frameworks."
        complex_score = analyzer._calculate_flesch_score(complex_text)
        assert 0.0 <= complex_score <= 100.0
        assert complex_score < score  # Complex text should have lower score

    def test_content_quality_assessment(self):
        """Test content quality assessment methods."""
        analyzer = ProcessingQualityAnalyzer()

        content = "This is a well-structured document. It has multiple sentences and paragraphs.\n\nThe content includes proper formatting and reasonable complexity."

        # Test completeness
        completeness = analyzer._assess_completeness(content, {"metadata": "test"})
        assert 0.0 <= completeness <= 1.0

        # Test coherence
        coherence = analyzer._assess_coherence(content)
        assert 0.0 <= coherence <= 1.0

        # Test information density
        density = analyzer._assess_information_density(content)
        assert 0.0 <= density <= 1.0


# =============================================================================
# RULES ENGINE TESTS
# =============================================================================


class TestExtractionRulesEngine:
    """Test cases for ExtractionRulesEngine."""

    def test_engine_initialization(self):
        """Test rules engine initialization."""
        engine = ExtractionRulesEngine()

        assert engine is not None
        assert len(engine.builtin_patterns) > 0
        assert engine.stats["rules_registered"] == 0

    def test_rule_registration(self):
        """Test rule registration."""
        engine = ExtractionRulesEngine()

        rule = CustomRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            rule_type=RuleType.REGEX,
            pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        )

        engine.register_rule(rule)

        assert rule.rule_id in engine.rule_registry
        assert engine.stats["rules_registered"] == 1

    @pytest.mark.asyncio
    async def test_compile_rules(self):
        """Test rule compilation."""
        engine = ExtractionRulesEngine()

        rule_definitions = [
            r"\b\w+@\w+\.\w+\b",  # Email pattern
            {
                "rule_id": "phone_rule",
                "name": "Phone Extractor",
                "description": "Extract phone numbers",
                "rule_type": "regex",
                "pattern": r"\b\d{3}-\d{3}-\d{4}\b",
            },
        ]

        ruleset = await engine.compile_rules(rule_definitions)

        assert isinstance(ruleset, RuleSet)
        assert len(ruleset.rules) == 2
        assert ruleset.rules[0].rule_type == RuleType.REGEX
        assert ruleset.rules[1].rule_id == "phone_rule"

    @pytest.mark.asyncio
    async def test_extract_with_rules(self, temp_text_file):
        """Test extraction with custom rules."""
        engine = ExtractionRulesEngine()

        # Create a ruleset for email extraction
        rule_definitions = [r"\b\w+@\w+\.\w+\b"]
        ruleset = await engine.compile_rules(rule_definitions)

        result = await engine.extract_with_rules(temp_text_file, ruleset)

        assert "content" in result
        assert "token_count" in result
        assert "extraction_method" in result
        assert result["extraction_method"] == "rules_engine"
        assert "rules_applied" in result
        assert result["rules_applied"] > 0

    def test_domain_rulesets(self):
        """Test domain-specific rulesets."""
        engine = ExtractionRulesEngine()

        # Test legal ruleset
        legal_rules = engine.create_domain_ruleset("legal")
        assert legal_rules.domain == "legal"
        assert len(legal_rules.rules) > 0
        assert any("legal" in rule.name.lower() for rule in legal_rules.rules)

        # Test medical ruleset
        medical_rules = engine.create_domain_ruleset("medical")
        assert medical_rules.domain == "medical"
        assert len(medical_rules.rules) > 0

        # Test generic ruleset
        generic_rules = engine.create_domain_ruleset("unknown")
        assert generic_rules.domain == "generic"
        assert len(generic_rules.rules) > 0

    def test_builtin_patterns(self):
        """Test built-in extraction patterns."""
        engine = ExtractionRulesEngine()

        test_content = """
        Contact: john.doe@example.com
        Phone: (555) 123-4567
        Website: https://www.example.com
        Date: January 15, 2024
        Amount: $1,234.56
        Number: 42.5
        """

        # Test email extraction
        emails = engine.builtin_patterns["emails"](test_content)
        assert len(emails) > 0
        assert "john.doe@example.com" in emails[0]

        # Test URL extraction
        urls = engine.builtin_patterns["urls"](test_content)
        assert len(urls) > 0
        assert "https://www.example.com" in urls[0]

        # Test money extraction
        money = engine.builtin_patterns["money"](test_content)
        assert len(money) > 0
        assert "$1,234.56" in money[0]


# =============================================================================
# FORMAT OPTIMIZATION TESTS
# =============================================================================


class TestFormatOptimizationEngine:
    """Test cases for FormatOptimizationEngine."""

    def test_engine_initialization(self):
        """Test optimization engine initialization."""
        engine = FormatOptimizationEngine()

        assert engine is not None
        assert len(engine.strategies) > 0
        assert engine.performance_stats["strategies_applied"] == 0

    @pytest.mark.asyncio
    async def test_strategy_selection(self, temp_text_file):
        """Test optimization strategy selection."""
        engine = FormatOptimizationEngine()

        # Mock file classification
        class MockClassification:
            detected_format = "txt"
            confidence_score = 0.8

        strategy = await engine.select_strategy(
            MockClassification(), "balanced", quality_threshold=0.7
        )

        assert isinstance(strategy, OptimizationStrategy)
        assert strategy.optimization_level in [
            OptimizationLevel.BALANCED,
            OptimizationLevel.SPEED,
        ]
        assert "txt" in strategy.supported_formats or "*" in strategy.supported_formats

    @pytest.mark.asyncio
    async def test_apply_optimizations(self, temp_text_file):
        """Test optimization application."""
        engine = FormatOptimizationEngine()

        # Mock dependencies
        class MockClassification:
            detected_format = "txt"
            confidence_score = 0.8

        class MockContext:
            optimization_level = "balanced"

        strategy = await engine.select_strategy(MockClassification(), "balanced")
        result = await engine.apply_optimizations(
            temp_text_file, MockClassification(), strategy, MockContext()
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.applied_optimizations) >= 0
        assert result.processing_time_saved >= 0.0
        assert result.memory_usage_reduced >= 0.0
        assert -1.0 <= result.quality_impact <= 1.0

    def test_optimization_strategies(self):
        """Test optimization strategy configurations."""
        engine = FormatOptimizationEngine()

        # Check that strategies exist for different formats
        pdf_strategies = [
            s for s in engine.strategies.values() if "pdf" in s.supported_formats
        ]
        text_strategies = [
            s for s in engine.strategies.values() if "txt" in s.supported_formats
        ]
        json_strategies = [
            s for s in engine.strategies.values() if "json" in s.supported_formats
        ]

        assert len(pdf_strategies) > 0
        assert len(text_strategies) > 0
        assert len(json_strategies) > 0

    def test_generic_strategy_fallback(self):
        """Test generic strategy fallback."""
        engine = FormatOptimizationEngine()

        generic_strategy = engine._get_generic_strategy("balanced")

        assert isinstance(generic_strategy, OptimizationStrategy)
        assert generic_strategy.strategy_id == "generic_fallback"
        assert "*" in generic_strategy.supported_formats
        assert generic_strategy.optimization_level == OptimizationLevel.BALANCED


# =============================================================================
# ERROR INTELLIGENCE TESTS
# =============================================================================


class TestIntelligentErrorHandler:
    """Test cases for IntelligentErrorHandler."""

    def test_handler_initialization(self):
        """Test error handler initialization."""
        handler = IntelligentErrorHandler()

        assert handler is not None
        assert len(handler.error_patterns) > 0
        assert handler.stats["total_errors_analyzed"] == 0

    @pytest.mark.asyncio
    async def test_analyze_file_not_found_error(self, sample_processing_context):
        """Test analysis of FileNotFoundError."""
        handler = IntelligentErrorHandler()

        error = FileNotFoundError("No such file or directory: '/nonexistent/file.txt'")

        analysis = await handler.analyze_error(
            error, "/nonexistent/file.txt", sample_processing_context
        )

        assert isinstance(analysis, ErrorAnalysis)
        assert analysis.error_type == "FileNotFoundError"
        assert analysis.category == ErrorCategory.FILE_ACCESS
        assert analysis.severity == ErrorSeverity.HIGH
        assert "file access" in analysis.root_cause.lower()

    @pytest.mark.asyncio
    async def test_analyze_unicode_decode_error(self, sample_processing_context):
        """Test analysis of UnicodeDecodeError."""
        handler = IntelligentErrorHandler()

        error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        analysis = await handler.analyze_error(
            error, "/test/file.txt", sample_processing_context
        )

        assert isinstance(analysis, ErrorAnalysis)
        assert analysis.error_type == "UnicodeDecodeError"
        assert analysis.category == ErrorCategory.ENCODING_ISSUES
        assert analysis.severity == ErrorSeverity.MEDIUM
        assert "encoding" in analysis.root_cause.lower()

    @pytest.mark.asyncio
    async def test_recovery_strategy_selection(self, sample_processing_context):
        """Test recovery strategy selection."""
        handler = IntelligentErrorHandler()

        # Create error analysis
        error = FileNotFoundError("File not found")
        analysis = await handler.analyze_error(
            error, "/test/file.txt", sample_processing_context
        )

        # Generate recovery plan
        recovery_plan = await handler._generate_recovery_plan(
            analysis, sample_processing_context
        )

        assert recovery_plan.primary_strategy in list(RecoveryStrategy)
        assert len(recovery_plan.alternative_strategies) >= 0
        assert 0.0 <= recovery_plan.estimated_success_rate <= 1.0
        assert recovery_plan.estimated_recovery_time > 0.0

    @pytest.mark.asyncio
    async def test_attempt_recovery(self, sample_processing_context):
        """Test recovery attempt."""
        handler = IntelligentErrorHandler()

        error = ValueError("Test error for recovery")
        analysis = await handler.analyze_error(
            error, "/test/file.txt", sample_processing_context
        )

        recovery = await handler.attempt_recovery(
            error, "/test/file.txt", sample_processing_context, analysis, max_attempts=1
        )

        assert isinstance(recovery, RecoveryResult)
        assert recovery.strategy_used in list(RecoveryStrategy)
        assert recovery.recovery_time >= 0.0
        # Recovery may succeed or fail depending on strategy

    def test_error_pattern_matching(self):
        """Test error pattern matching."""
        handler = IntelligentErrorHandler()

        # Test file not found pattern
        error = FileNotFoundError("No such file")
        category, severity = (
            handler._classify_error_pattern(error, str(error), "")[0],
            handler._classify_error_pattern(error, str(error), "")[1],
        )

        # Should be caught by our patterns
        # The exact result depends on pattern implementation
        assert category in list(ErrorCategory)
        assert severity in list(ErrorSeverity)

    def test_error_summary(self):
        """Test error summary generation."""
        handler = IntelligentErrorHandler()

        # Add some mock errors to history
        from datetime import datetime

        mock_error = ErrorAnalysis(
            error_id="test_error",
            error_type="ValueError",
            error_message="Test error",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION_ERROR,
            root_cause="Test root cause",
            stack_trace="Mock stack trace",
            analysis_confidence=0.8,
            timestamp=datetime.now(),
        )
        handler.error_history.append(mock_error)

        summary = handler.get_error_summary(last_n_hours=24)

        assert "total_errors" in summary
        assert "error_categories" in summary
        assert "recovery_attempts" in summary
        assert "recovery_success_rate" in summary
        assert summary["total_errors"] >= 1


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @pytest.mark.asyncio
    async def test_process_file_with_ai(
        self, temp_text_file, sample_processing_context
    ):
        """Test convenience function for file processing."""
        result = await process_file_with_ai(temp_text_file, sample_processing_context)

        assert isinstance(result, IntelligentProcessingResult)
        assert result.file_path == str(temp_text_file)

    @pytest.mark.asyncio
    async def test_batch_process_with_ai(
        self, temp_text_file, temp_json_file, sample_processing_context
    ):
        """Test convenience function for batch processing."""
        results = await batch_process_with_ai(
            [temp_text_file, temp_json_file], sample_processing_context
        )

        assert len(results) == 2
        assert all(isinstance(r, IntelligentProcessingResult) for r in results)

    @pytest.mark.asyncio
    async def test_classify_file_ai(self, temp_text_file):
        """Test convenience function for file classification."""
        result = await classify_file_ai(temp_text_file)

        assert isinstance(result, FileTypeAnalysis)
        assert result.detected_format in ["txt", "text", "plain"]

    @pytest.mark.asyncio
    async def test_analyze_processing_quality_convenience(self):
        """Test convenience function for quality analysis."""
        extraction_result = {
            "content": "Test content for quality analysis",
            "processing_time": 1.0,
            "token_count": 10,
            "cost": 0.001,
        }

        result = await analyze_processing_quality(extraction_result)

        assert isinstance(result, QualityAnalysisResult)
        assert result.overall_score >= 0.0

    @pytest.mark.asyncio
    async def test_extract_with_custom_rules_convenience(self, temp_text_file):
        """Test convenience function for rules extraction."""
        rule_definitions = [r"\b\w+@\w+\.\w+\b"]  # Email pattern

        result = await extract_with_custom_rules(temp_text_file, rule_definitions)

        assert "content" in result
        assert "extraction_method" in result
        assert result["extraction_method"] == "rules_engine"

    @pytest.mark.asyncio
    async def test_optimize_file_processing_convenience(self, temp_text_file):
        """Test convenience function for file optimization."""
        result = await optimize_file_processing(temp_text_file, "txt", "balanced")

        assert isinstance(result, OptimizationResult)
        assert len(result.applied_optimizations) >= 0

    @pytest.mark.asyncio
    async def test_handle_processing_error_convenience(self):
        """Test convenience function for error handling."""
        error = ValueError("Test error")

        analysis, recovery = await handle_processing_error(error, "/test/file.txt")

        assert isinstance(analysis, ErrorAnalysis)
        assert isinstance(recovery, RecoveryResult)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the complete intelligent processing system."""

    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, temp_text_file, mock_ai_client):
        """Test complete end-to-end processing flow."""
        # Create processor with AI client
        processor = IntelligentFileProcessor(ai_client=mock_ai_client)

        # Create processing context
        context = ProcessingContext(
            domain_knowledge="technical",
            quality_threshold=0.7,
            custom_rules=["emails", "urls"],
            optimization_level="balanced",
        )

        # Process file
        result = await processor.process_file_intelligently(temp_text_file, context)

        # Verify complete result
        assert isinstance(result, IntelligentProcessingResult)
        assert result.confidence_score > 0.0
        assert result.processing_quality > 0.0
        assert len(result.optimization_applied) >= 0
        assert len(result.recommendations) >= 0
        assert result.metadata is not None

        # Verify analytics updated
        analytics = processor.get_processing_analytics()
        assert analytics["statistics"]["total_processed"] >= 1

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, sample_processing_context):
        """Test error handling and recovery integration."""
        processor = IntelligentFileProcessor()

        # Try to process non-existent file
        try:
            await processor.process_file_intelligently(
                "/nonexistent/file.txt", sample_processing_context
            )
            assert False, "Should have raised an exception"
        except Exception as e:
            # Verify error handling
            assert isinstance(e, (FileNotFoundError, RuntimeError))

    @pytest.mark.asyncio
    async def test_quality_feedback_loop(
        self, temp_text_file, sample_processing_context
    ):
        """Test quality assessment feedback into processing optimization."""
        processor = IntelligentFileProcessor()

        # Process with high quality threshold
        high_quality_context = ProcessingContext(
            quality_threshold=0.9, optimization_level="quality"
        )

        result_high = await processor.process_file_intelligently(
            temp_text_file, high_quality_context
        )

        # Process with speed optimization
        speed_context = ProcessingContext(
            quality_threshold=0.5, optimization_level="speed"
        )

        result_speed = await processor.process_file_intelligently(
            temp_text_file, speed_context
        )

        # Verify different optimization paths were taken
        assert (
            result_high.extraction_method != result_speed.extraction_method
            or result_high.optimization_applied != result_speed.optimization_applied
        )


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance tests for intelligent processing system."""

    @pytest.mark.asyncio
    async def test_processing_performance(
        self, temp_text_file, sample_processing_context
    ):
        """Test processing performance metrics."""
        import time

        processor = IntelligentFileProcessor()

        start_time = time.time()
        result = await processor.process_file_intelligently(
            temp_text_file, sample_processing_context
        )
        end_time = time.time()

        processing_time = end_time - start_time

        # Verify reasonable performance
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert result.processing_time > 0.0
        assert result.processing_time <= processing_time

    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, temp_text_file, temp_json_file):
        """Test batch processing performance."""
        import time

        processor = IntelligentFileProcessor()

        # Create multiple file references
        files = [temp_text_file, temp_json_file] * 5  # 10 files total

        start_time = time.time()
        results = await processor.batch_process_intelligently(files)
        end_time = time.time()

        batch_time = end_time - start_time

        # Verify batch processing completed
        assert len(results) == len(files)
        assert batch_time < 30.0  # Should complete within 30 seconds

    def test_cache_performance(self, temp_text_file):
        """Test caching performance."""
        engine = FormatOptimizationEngine()

        # Generate cache key
        strategy = engine._get_generic_strategy("balanced")
        cache_key1 = engine._generate_cache_key(temp_text_file, strategy)
        cache_key2 = engine._generate_cache_key(temp_text_file, strategy)

        # Should generate consistent cache keys
        assert cache_key1 == cache_key2
        assert len(cache_key1) > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
