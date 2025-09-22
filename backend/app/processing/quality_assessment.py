"""
Processing Quality Assessment for brAIn v2.0

This module provides comprehensive quality assessment for file processing results,
including content quality, extraction accuracy, and processing efficiency metrics.

Author: BMad Team
"""

import re
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

# Core imports
from pydantic import BaseModel, Field
import anthropic
import openai
from langfuse import Langfuse

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ExtractionQualityMetrics(BaseModel):
    """Metrics for extraction quality assessment."""
    completeness_score: float = Field(ge=0.0, le=1.0, description="Content completeness")
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Extraction accuracy")
    structure_preservation: float = Field(ge=0.0, le=1.0, description="Structure preservation")
    metadata_extraction: float = Field(ge=0.0, le=1.0, description="Metadata extraction quality")
    character_encoding: float = Field(ge=0.0, le=1.0, description="Character encoding quality")

class ContentQualityMetrics(BaseModel):
    """Metrics for content quality assessment."""
    readability_score: float = Field(ge=0.0, le=100.0, description="Flesch reading ease score")
    coherence_score: float = Field(ge=0.0, le=1.0, description="Content coherence")
    information_density: float = Field(ge=0.0, le=1.0, description="Information density")
    language_consistency: float = Field(ge=0.0, le=1.0, description="Language consistency")
    formatting_quality: float = Field(ge=0.0, le=1.0, description="Formatting quality")

class ProcessingEfficiencyMetrics(BaseModel):
    """Metrics for processing efficiency assessment."""
    speed_score: float = Field(ge=0.0, le=1.0, description="Processing speed score")
    resource_utilization: float = Field(ge=0.0, le=1.0, description="Resource utilization")
    error_rate: float = Field(ge=0.0, le=1.0, description="Processing error rate")
    cost_efficiency: float = Field(ge=0.0, le=1.0, description="Cost efficiency score")
    scalability_score: float = Field(ge=0.0, le=1.0, description="Scalability assessment")

class QualityAnalysisResult(BaseModel):
    """Comprehensive quality analysis result."""
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    processing_efficiency: float = Field(ge=0.0, le=1.0, description="Processing efficiency")
    extraction_metrics: ExtractionQualityMetrics
    content_metrics: ContentQualityMetrics
    efficiency_metrics: ProcessingEfficiencyMetrics
    recommendations: List[str] = Field(default_factory=list)
    quality_issues: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Additional metrics")
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# QUALITY ASSESSMENT ENGINE
# =============================================================================

class ProcessingQualityAnalyzer:
    """
    Comprehensive quality analyzer for processing results using multiple metrics
    and AI-powered assessment techniques.
    """
    
    def __init__(
        self,
        ai_client: Optional[Union[anthropic.Anthropic, openai.OpenAI]] = None,
        langfuse_client: Optional[Langfuse] = None
    ):
        """
        Initialize quality analyzer.
        
        Args:
            ai_client: AI client for intelligent quality assessment
            langfuse_client: Langfuse for monitoring
        """
        self.ai_client = ai_client
        self.langfuse = langfuse_client
        
        # Quality thresholds
        self.thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.6,
            "poor": 0.4
        }
        
        # Analysis statistics
        self.stats = {
            "analyses_performed": 0,
            "high_quality_results": 0,
            "quality_improvements_suggested": 0,
            "ai_assessments_completed": 0
        }
    
    async def analyze_processing_quality(
        self,
        extraction_result: Dict[str, Any],
        file_classification: Any,  # FileTypeAnalysis from file_classification.py
        processing_context: Any   # ProcessingContext from intelligent_processor.py
    ) -> QualityAnalysisResult:
        """
        Perform comprehensive quality analysis of processing results.
        
        Args:
            extraction_result: Results from file processing
            file_classification: File classification results
            processing_context: Processing context and preferences
            
        Returns:
            Comprehensive quality analysis
        """
        start_time = datetime.now()
        
        try:
            # Extract content for analysis
            content = extraction_result.get("content", "")
            processing_time = extraction_result.get("processing_time", 0.0)
            token_count = extraction_result.get("token_count", 0)
            cost = float(extraction_result.get("cost", 0.0))
            
            # Step 1: Assess extraction quality
            extraction_metrics = await self._assess_extraction_quality(
                extraction_result, file_classification
            )
            
            # Step 2: Assess content quality
            content_metrics = await self._assess_content_quality(content)
            
            # Step 3: Assess processing efficiency
            efficiency_metrics = await self._assess_processing_efficiency(
                processing_time, token_count, cost, file_classification
            )
            
            # Step 4: AI-powered quality assessment (if available)
            ai_quality_insights = {}
            if self.ai_client and content:
                ai_quality_insights = await self._ai_quality_assessment(
                    content, extraction_result, file_classification
                )
            
            # Step 5: Calculate overall score
            overall_score = self._calculate_overall_score(
                extraction_metrics, content_metrics, efficiency_metrics
            )
            
            # Step 6: Generate recommendations
            recommendations = await self._generate_quality_recommendations(
                extraction_metrics, content_metrics, efficiency_metrics, ai_quality_insights
            )
            
            # Step 7: Identify quality issues
            quality_issues = await self._identify_quality_issues(
                extraction_metrics, content_metrics, efficiency_metrics
            )
            
            # Calculate processing efficiency
            analysis_time = (datetime.now() - start_time).total_seconds()
            processing_efficiency = min(1.0 / (analysis_time + 0.1), 1.0)
            
            # Create comprehensive result
            result = QualityAnalysisResult(
                overall_score=overall_score,
                processing_efficiency=processing_efficiency,
                extraction_metrics=extraction_metrics,
                content_metrics=content_metrics,
                efficiency_metrics=efficiency_metrics,
                recommendations=recommendations,
                quality_issues=quality_issues,
                metrics={
                    "content_length": len(content),
                    "token_count": token_count,
                    "processing_cost": cost,
                    "analysis_time": analysis_time,
                    "ai_insights": ai_quality_insights
                },
                analysis_metadata={
                    "analyzer_version": "2.0",
                    "analysis_timestamp": start_time.isoformat(),
                    "file_format": getattr(file_classification, 'detected_format', 'unknown'),
                    "confidence_score": getattr(file_classification, 'confidence_score', 0.0)
                }
            )
            
            # Update statistics
            self.stats["analyses_performed"] += 1
            if overall_score >= self.thresholds["good"]:
                self.stats["high_quality_results"] += 1
            if recommendations:
                self.stats["quality_improvements_suggested"] += 1
            if ai_quality_insights:
                self.stats["ai_assessments_completed"] += 1
            
            return result
            
        except Exception as e:
            # Return basic assessment on error
            return QualityAnalysisResult(
                overall_score=0.5,
                processing_efficiency=0.5,
                extraction_metrics=ExtractionQualityMetrics(
                    completeness_score=0.5,
                    accuracy_score=0.5,
                    structure_preservation=0.5,
                    metadata_extraction=0.5,
                    character_encoding=0.5
                ),
                content_metrics=ContentQualityMetrics(
                    readability_score=50.0,
                    coherence_score=0.5,
                    information_density=0.5,
                    language_consistency=0.5,
                    formatting_quality=0.5
                ),
                efficiency_metrics=ProcessingEfficiencyMetrics(
                    speed_score=0.5,
                    resource_utilization=0.5,
                    error_rate=0.5,
                    cost_efficiency=0.5,
                    scalability_score=0.5
                ),
                recommendations=[f"Quality analysis failed: {str(e)}"],
                quality_issues=[{"type": "analysis_error", "description": str(e)}]
            )
    
    async def _assess_extraction_quality(
        self,
        extraction_result: Dict[str, Any],
        file_classification: Any
    ) -> ExtractionQualityMetrics:
        """Assess the quality of content extraction."""
        content = extraction_result.get("content", "")
        metadata = extraction_result.get("metadata", {})
        
        # Completeness assessment
        completeness_score = self._assess_completeness(content, extraction_result)
        
        # Accuracy assessment
        accuracy_score = self._assess_accuracy(content, file_classification)
        
        # Structure preservation
        structure_preservation = self._assess_structure_preservation(content, extraction_result)
        
        # Metadata extraction quality
        metadata_extraction = self._assess_metadata_extraction(metadata, file_classification)
        
        # Character encoding quality
        character_encoding = self._assess_character_encoding(content)
        
        return ExtractionQualityMetrics(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            structure_preservation=structure_preservation,
            metadata_extraction=metadata_extraction,
            character_encoding=character_encoding
        )
    
    async def _assess_content_quality(self, content: str) -> ContentQualityMetrics:
        """Assess the quality of extracted content."""
        if not content:
            return ContentQualityMetrics(
                readability_score=0.0,
                coherence_score=0.0,
                information_density=0.0,
                language_consistency=0.0,
                formatting_quality=0.0
            )
        
        # Readability score using Flesch formula
        readability_score = self._calculate_flesch_score(content)
        
        # Coherence score
        coherence_score = self._assess_coherence(content)
        
        # Information density
        information_density = self._assess_information_density(content)
        
        # Language consistency
        language_consistency = self._assess_language_consistency(content)
        
        # Formatting quality
        formatting_quality = self._assess_formatting_quality(content)
        
        return ContentQualityMetrics(
            readability_score=readability_score,
            coherence_score=coherence_score,
            information_density=information_density,
            language_consistency=language_consistency,
            formatting_quality=formatting_quality
        )
    
    async def _assess_processing_efficiency(
        self,
        processing_time: float,
        token_count: int,
        cost: float,
        file_classification: Any
    ) -> ProcessingEfficiencyMetrics:
        """Assess processing efficiency metrics."""
        
        # Speed score (inversely related to processing time)
        expected_time = self._estimate_expected_processing_time(token_count, file_classification)
        speed_score = min(expected_time / max(processing_time, 0.1), 1.0)
        
        # Resource utilization (based on token efficiency)
        resource_utilization = self._assess_resource_utilization(token_count, processing_time)
        
        # Error rate (assumed low if we got here)
        error_rate = 0.1  # Base error rate
        
        # Cost efficiency
        expected_cost = self._estimate_expected_cost(token_count)
        cost_efficiency = min(expected_cost / max(cost, 0.001), 1.0) if cost > 0 else 1.0
        
        # Scalability score
        scalability_score = self._assess_scalability(processing_time, token_count)
        
        return ProcessingEfficiencyMetrics(
            speed_score=speed_score,
            resource_utilization=resource_utilization,
            error_rate=error_rate,
            cost_efficiency=cost_efficiency,
            scalability_score=scalability_score
        )
    
    async def _ai_quality_assessment(
        self,
        content: str,
        extraction_result: Dict[str, Any],
        file_classification: Any
    ) -> Dict[str, Any]:
        """Use AI for advanced quality assessment."""
        if not self.ai_client:
            return {}
        
        try:
            # Create sample for AI analysis (first 1000 chars)
            content_sample = content[:1000]
            
            prompt = f"""
            Assess the quality of this extracted content from a document processing operation:
            
            Original file type: {getattr(file_classification, 'detected_format', 'unknown')}
            Content sample:
            {content_sample}
            
            Please evaluate:
            1. Content clarity and readability (1-10)
            2. Information completeness (1-10)
            3. Structural organization (1-10)
            4. Language quality and coherence (1-10)
            5. Overall extraction quality (1-10)
            6. Specific issues or concerns
            7. Suggestions for improvement
            
            Format as JSON with these exact keys: clarity, completeness, structure, language, overall, issues, suggestions
            """
            
            # Call AI service
            if isinstance(self.ai_client, anthropic.Anthropic):
                response = await self._call_anthropic_quality_assessment(prompt)
            elif isinstance(self.ai_client, openai.OpenAI):
                response = await self._call_openai_quality_assessment(prompt)
            else:
                response = {}
            
            return response
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _call_anthropic_quality_assessment(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic for quality assessment."""
        try:
            message = self.ai_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=800,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            # Try to parse JSON
            import json
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"raw_response": response_text}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _call_openai_quality_assessment(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI for quality assessment."""
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0
            )
            
            response_text = response.choices[0].message.content
            
            # Try to parse JSON
            import json
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"raw_response": response_text}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_overall_score(
        self,
        extraction_metrics: ExtractionQualityMetrics,
        content_metrics: ContentQualityMetrics,
        efficiency_metrics: ProcessingEfficiencyMetrics
    ) -> float:
        """Calculate overall quality score from all metrics."""
        
        # Weights for different metric categories
        extraction_weight = 0.4
        content_weight = 0.4
        efficiency_weight = 0.2
        
        # Calculate weighted averages
        extraction_avg = (
            extraction_metrics.completeness_score +
            extraction_metrics.accuracy_score +
            extraction_metrics.structure_preservation +
            extraction_metrics.metadata_extraction +
            extraction_metrics.character_encoding
        ) / 5
        
        content_avg = (
            min(content_metrics.readability_score / 100.0, 1.0) +
            content_metrics.coherence_score +
            content_metrics.information_density +
            content_metrics.language_consistency +
            content_metrics.formatting_quality
        ) / 5
        
        efficiency_avg = (
            efficiency_metrics.speed_score +
            efficiency_metrics.resource_utilization +
            (1.0 - efficiency_metrics.error_rate) +  # Invert error rate
            efficiency_metrics.cost_efficiency +
            efficiency_metrics.scalability_score
        ) / 5
        
        # Calculate final weighted score
        overall_score = (
            extraction_avg * extraction_weight +
            content_avg * content_weight +
            efficiency_avg * efficiency_weight
        )
        
        return min(overall_score, 1.0)
    
    async def _generate_quality_recommendations(
        self,
        extraction_metrics: ExtractionQualityMetrics,
        content_metrics: ContentQualityMetrics,
        efficiency_metrics: ProcessingEfficiencyMetrics,
        ai_insights: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable quality recommendations."""
        recommendations = []
        
        # Extraction-based recommendations
        if extraction_metrics.completeness_score < 0.8:
            recommendations.append("Consider using format-specific extraction methods for better completeness")
        
        if extraction_metrics.accuracy_score < 0.7:
            recommendations.append("Review extraction accuracy - may need OCR improvement or custom rules")
        
        if extraction_metrics.structure_preservation < 0.6:
            recommendations.append("Enable structure-aware processing to preserve document layout")
        
        # Content-based recommendations
        if content_metrics.readability_score < 30:
            recommendations.append("Content has low readability - consider preprocessing or formatting")
        
        if content_metrics.coherence_score < 0.6:
            recommendations.append("Content lacks coherence - check for extraction errors or fragmentation")
        
        if content_metrics.information_density < 0.4:
            recommendations.append("Low information density - may need filtering or summarization")
        
        # Efficiency-based recommendations
        if efficiency_metrics.speed_score < 0.5:
            recommendations.append("Processing speed is slow - consider optimization or batch processing")
        
        if efficiency_metrics.cost_efficiency < 0.6:
            recommendations.append("High processing cost - review token usage and model selection")
        
        # AI-based recommendations
        if ai_insights.get("suggestions"):
            ai_suggestions = ai_insights["suggestions"]
            if isinstance(ai_suggestions, list):
                recommendations.extend(ai_suggestions[:3])  # Limit to top 3
            elif isinstance(ai_suggestions, str):
                recommendations.append(ai_suggestions)
        
        return recommendations[:10]  # Limit total recommendations
    
    async def _identify_quality_issues(
        self,
        extraction_metrics: ExtractionQualityMetrics,
        content_metrics: ContentQualityMetrics,
        efficiency_metrics: ProcessingEfficiencyMetrics
    ) -> List[Dict[str, Any]]:
        """Identify specific quality issues."""
        issues = []
        
        # Critical issues (score < 0.4)
        if extraction_metrics.completeness_score < 0.4:
            issues.append({
                "type": "critical",
                "category": "extraction",
                "description": "Very low content completeness detected",
                "score": extraction_metrics.completeness_score,
                "impact": "high"
            })
        
        if content_metrics.coherence_score < 0.4:
            issues.append({
                "type": "critical",
                "category": "content",
                "description": "Content lacks coherence and structure",
                "score": content_metrics.coherence_score,
                "impact": "high"
            })
        
        # Warning issues (score < 0.6)
        if extraction_metrics.accuracy_score < 0.6:
            issues.append({
                "type": "warning",
                "category": "extraction",
                "description": "Moderate accuracy concerns in extraction",
                "score": extraction_metrics.accuracy_score,
                "impact": "medium"
            })
        
        if efficiency_metrics.cost_efficiency < 0.6:
            issues.append({
                "type": "warning",
                "category": "efficiency",
                "description": "Higher than expected processing costs",
                "score": efficiency_metrics.cost_efficiency,
                "impact": "medium"
            })
        
        return issues
    
    # Helper methods for quality assessment
    
    def _assess_completeness(self, content: str, extraction_result: Dict[str, Any]) -> float:
        """Assess content completeness."""
        if not content:
            return 0.0
        
        # Basic completeness indicators
        indicators = {
            "has_content": len(content) > 50,
            "has_structure": bool(re.search(r'\n\s*\n', content)),
            "balanced_length": 50 < len(content) < 100000,
            "no_truncation": not content.endswith('...'),
            "minimal_errors": content.count('�') < len(content) * 0.01
        }
        
        return sum(indicators.values()) / len(indicators)
    
    def _assess_accuracy(self, content: str, file_classification: Any) -> float:
        """Assess extraction accuracy."""
        if not content:
            return 0.0
        
        # Accuracy indicators based on file type
        base_accuracy = 0.8  # Base accuracy assumption
        
        # Adjust based on classification confidence
        confidence_factor = getattr(file_classification, 'confidence_score', 0.5)
        
        # Look for extraction artifacts that indicate low accuracy
        artifacts = [
            len(re.findall(r'[^\x00-\x7F]', content)) / len(content),  # Non-ASCII ratio
            content.count('�') / max(len(content), 1),  # Encoding errors
            len(re.findall(r'\s{5,}', content)) / max(len(content.split()), 1)  # Excessive whitespace
        ]
        
        artifact_penalty = sum(artifacts) / len(artifacts)
        
        return max(base_accuracy * confidence_factor - artifact_penalty, 0.0)
    
    def _assess_structure_preservation(self, content: str, extraction_result: Dict[str, Any]) -> float:
        """Assess how well document structure was preserved."""
        if not content:
            return 0.0
        
        structure_indicators = {
            "has_paragraphs": len(content.split('\n\n')) > 1,
            "has_headers": bool(re.search(r'^.{1,100}$', content, re.MULTILINE)),
            "has_lists": bool(re.search(r'^\s*[-*•]\s+', content, re.MULTILINE)),
            "proper_spacing": not bool(re.search(r'\S{200,}', content)),
            "line_breaks": '\n' in content
        }
        
        return sum(structure_indicators.values()) / len(structure_indicators)
    
    def _assess_metadata_extraction(self, metadata: Dict[str, Any], file_classification: Any) -> float:
        """Assess metadata extraction quality."""
        if not metadata:
            return 0.2  # Some metadata is better than none
        
        valuable_fields = ['title', 'author', 'creation_date', 'modification_date', 'subject']
        present_fields = sum(1 for field in valuable_fields if metadata.get(field))
        
        return present_fields / len(valuable_fields)
    
    def _assess_character_encoding(self, content: str) -> float:
        """Assess character encoding quality."""
        if not content:
            return 1.0
        
        # Count encoding-related issues
        total_chars = len(content)
        encoding_errors = content.count('�')  # Replacement character
        non_printable = len(re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', content))
        
        if total_chars == 0:
            return 1.0
        
        error_ratio = (encoding_errors + non_printable) / total_chars
        return max(1.0 - error_ratio * 10, 0.0)  # Penalize encoding errors
    
    def _calculate_flesch_score(self, content: str) -> float:
        """Calculate Flesch reading ease score."""
        if not content or len(content) < 10:
            return 50.0  # Neutral score for very short content
        
        # Basic sentence and syllable counting
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        
        if sentences == 0 or words == 0:
            return 50.0
        
        # Rough syllable estimation (vowel groups)
        syllables = len(re.findall(r'[aeiouyAEIOUY]+', content))
        
        if syllables == 0:
            syllables = words  # Fallback
        
        # Flesch formula
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        return max(0.0, min(100.0, score))
    
    def _assess_coherence(self, content: str) -> float:
        """Assess content coherence."""
        if not content or len(content) < 100:
            return 0.5
        
        # Simple coherence indicators
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 2:
            return 0.3
        
        # Check for transition words
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 
                          'nevertheless', 'additionally', 'finally', 'first', 'second']
        
        transition_count = sum(content.lower().count(word) for word in transition_words)
        transition_score = min(transition_count / max(len(paragraphs), 1) * 0.5, 1.0)
        
        # Check for consistent topic (repeated key terms)
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only count substantial words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            top_words = sorted(word_freq.values(), reverse=True)[:5]
            consistency_score = sum(top_words) / len(words)
        else:
            consistency_score = 0.0
        
        return (transition_score + min(consistency_score * 10, 1.0)) / 2
    
    def _assess_information_density(self, content: str) -> float:
        """Assess information density."""
        if not content:
            return 0.0
        
        # Count informative elements
        numbers = len(re.findall(r'\d+', content))
        capitals = len(re.findall(r'[A-Z][a-z]+', content))
        punctuation = len(re.findall(r'[.,:;!?]', content))
        words = len(content.split())
        
        if words == 0:
            return 0.0
        
        density_score = (numbers + capitals * 0.5 + punctuation * 0.2) / words
        return min(density_score, 1.0)
    
    def _assess_language_consistency(self, content: str) -> float:
        """Assess language consistency."""
        # Simple check for mixed languages or encoding issues
        if not content:
            return 1.0
        
        # Check for consistent character sets
        ascii_chars = len(re.findall(r'[a-zA-Z0-9\s]', content))
        total_chars = len(content)
        
        if total_chars == 0:
            return 1.0
        
        ascii_ratio = ascii_chars / total_chars
        
        # High ASCII ratio generally indicates consistent English text
        # Lower ratios might indicate mixed languages or encoding issues
        return min(ascii_ratio + 0.2, 1.0)
    
    def _assess_formatting_quality(self, content: str) -> float:
        """Assess formatting quality."""
        if not content:
            return 0.0
        
        formatting_indicators = {
            "proper_spacing": not bool(re.search(r'\S{100,}', content)),
            "paragraph_breaks": '\n\n' in content,
            "no_excessive_whitespace": not bool(re.search(r'\s{10,}', content)),
            "consistent_line_endings": content.count('\n') > content.count('\r\n'),
            "readable_structure": len(content.split('\n')) > 1
        }
        
        return sum(formatting_indicators.values()) / len(formatting_indicators)
    
    def _estimate_expected_processing_time(self, token_count: int, file_classification: Any) -> float:
        """Estimate expected processing time based on content."""
        # Base time per token (seconds)
        base_time_per_token = 0.001
        
        # Adjust based on file complexity
        complexity_multiplier = 1.0
        if hasattr(file_classification, 'detected_format'):
            format_type = file_classification.detected_format
            if format_type in ['pdf', 'docx', 'doc']:
                complexity_multiplier = 1.5
            elif format_type in ['txt', 'md']:
                complexity_multiplier = 0.8
        
        return token_count * base_time_per_token * complexity_multiplier
    
    def _assess_resource_utilization(self, token_count: int, processing_time: float) -> float:
        """Assess resource utilization efficiency."""
        if processing_time <= 0:
            return 1.0
        
        # Tokens processed per second
        throughput = token_count / processing_time
        
        # Expected throughput (tokens per second)
        expected_throughput = 1000  # Baseline expectation
        
        efficiency = min(throughput / expected_throughput, 1.0)
        return max(efficiency, 0.1)
    
    def _estimate_expected_cost(self, token_count: int) -> float:
        """Estimate expected processing cost."""
        # Base cost per token (rough estimate)
        cost_per_token = 0.0001
        return token_count * cost_per_token
    
    def _assess_scalability(self, processing_time: float, token_count: int) -> float:
        """Assess processing scalability."""
        if token_count <= 0:
            return 1.0
        
        # Time per token
        time_per_token = processing_time / token_count
        
        # Good scalability if time per token is low and consistent
        if time_per_token < 0.001:  # Less than 1ms per token
            return 1.0
        elif time_per_token < 0.01:  # Less than 10ms per token
            return 0.8
        elif time_per_token < 0.1:   # Less than 100ms per token
            return 0.6
        else:
            return 0.3
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get analyzer health status."""
        return {
            "ai_client_available": self.ai_client is not None,
            "langfuse_available": self.langfuse is not None,
            "statistics": self.stats.copy(),
            "thresholds": self.thresholds.copy()
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global analyzer instance
_default_analyzer = None

def get_default_quality_analyzer() -> ProcessingQualityAnalyzer:
    """Get default quality analyzer instance."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = ProcessingQualityAnalyzer()
    return _default_analyzer

async def analyze_processing_quality(
    extraction_result: Dict[str, Any],
    file_classification: Any = None,
    processing_context: Any = None
) -> QualityAnalysisResult:
    """Analyze processing quality with default analyzer."""
    analyzer = get_default_quality_analyzer()
    return await analyzer.analyze_processing_quality(
        extraction_result, file_classification, processing_context
    )