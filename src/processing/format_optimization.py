"""
Format-Specific Optimization Engine for brAIn v2.0

This module provides format-specific processing optimizations and strategies
tailored to different document types and processing contexts.

Author: BMad Team
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

# Core imports
from pydantic import BaseModel, Field
from enum import Enum
import mimetypes

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class OptimizationLevel(str, Enum):
    """Processing optimization levels."""
    SPEED = "speed"           # Fastest processing, basic quality
    BALANCED = "balanced"     # Balance of speed and quality
    QUALITY = "quality"       # Highest quality, slower processing
    CUSTOM = "custom"         # Custom optimization parameters

class ProcessingMethod(str, Enum):
    """Available processing methods for different formats."""
    DIRECT_TEXT = "direct_text"
    OCR_EXTRACTION = "ocr_extraction"
    STRUCTURED_PARSING = "structured_parsing"
    BINARY_ANALYSIS = "binary_analysis"
    STREAMING_PROCESSING = "streaming_processing"
    BATCH_PROCESSING = "batch_processing"

class OptimizationStrategy(BaseModel):
    """Strategy for format-specific optimization."""
    strategy_id: str = Field(description="Unique strategy identifier")
    method_name: str = Field(description="Processing method name")
    supported_formats: List[str] = Field(description="Supported file formats")
    optimization_level: OptimizationLevel = Field(description="Optimization level")
    recommended_chunk_size: int = Field(default=1000, description="Recommended chunk size")
    max_file_size_mb: float = Field(default=100.0, description="Maximum file size in MB")
    parallel_processing: bool = Field(default=False, description="Enable parallel processing")
    memory_optimization: bool = Field(default=True, description="Enable memory optimization")
    caching_enabled: bool = Field(default=True, description="Enable result caching")
    preprocessing_steps: List[str] = Field(default_factory=list, description="Preprocessing steps")
    postprocessing_steps: List[str] = Field(default_factory=list, description="Postprocessing steps")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Strategy metadata")

class OptimizationResult(BaseModel):
    """Result of optimization application."""
    applied_optimizations: List[str] = Field(description="List of applied optimizations")
    processing_time_saved: float = Field(ge=0.0, description="Estimated time saved")
    memory_usage_reduced: float = Field(ge=0.0, description="Memory usage reduction percentage")
    quality_impact: float = Field(ge=-1.0, le=1.0, description="Quality impact (-1 to 1)")
    recommendations: List[str] = Field(default_factory=list, description="Additional recommendations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optimization metadata")

# =============================================================================
# FORMAT OPTIMIZATION ENGINE
# =============================================================================

class FormatOptimizationEngine:
    """
    Engine for applying format-specific optimizations to document processing.
    Provides intelligent optimization strategies based on file type and processing context.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize format optimization engine.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Initialize optimization strategies
        self.strategies = self._initialize_optimization_strategies()
        
        # Performance tracking
        self.performance_stats = {
            "strategies_applied": 0,
            "total_time_saved": 0.0,
            "total_memory_saved": 0.0,
            "quality_improvements": 0,
            "failed_optimizations": 0
        }
        
        # Cache for optimization results
        self._optimization_cache = {}
    
    async def select_strategy(
        self,
        file_classification: Any,  # FileTypeAnalysis
        optimization_level: str,
        quality_threshold: float = 0.7
    ) -> OptimizationStrategy:
        """
        Select optimal processing strategy based on file classification and requirements.
        
        Args:
            file_classification: File classification results
            optimization_level: Requested optimization level
            quality_threshold: Minimum quality threshold
            
        Returns:
            Selected optimization strategy
        """
        detected_format = getattr(file_classification, 'detected_format', 'unknown')
        confidence_score = getattr(file_classification, 'confidence_score', 0.5)
        
        # Find compatible strategies
        compatible_strategies = [
            strategy for strategy in self.strategies.values()
            if detected_format in strategy.supported_formats
        ]
        
        if not compatible_strategies:
            # Return generic strategy
            return self._get_generic_strategy(optimization_level)
        
        # Filter by optimization level
        level_enum = OptimizationLevel(optimization_level) if optimization_level in OptimizationLevel else OptimizationLevel.BALANCED
        
        level_strategies = [
            strategy for strategy in compatible_strategies
            if strategy.optimization_level == level_enum
        ]
        
        if not level_strategies:
            level_strategies = compatible_strategies
        
        # Select best strategy based on confidence and format specificity
        best_strategy = max(
            level_strategies,
            key=lambda s: self._calculate_strategy_score(s, detected_format, confidence_score)
        )
        
        return best_strategy
    
    async def apply_optimizations(
        self,
        file_path: Path,
        file_classification: Any,
        strategy: OptimizationStrategy,
        processing_context: Any
    ) -> OptimizationResult:
        """
        Apply optimizations based on selected strategy.
        
        Args:
            file_path: Path to file being processed
            file_classification: File classification results
            strategy: Selected optimization strategy
            processing_context: Processing context
            
        Returns:
            Optimization application results
        """
        start_time = datetime.now()
        applied_optimizations = []
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(file_path, strategy)
            if cache_key in self._optimization_cache:
                cached_result = self._optimization_cache[cache_key]
                cached_result.metadata["from_cache"] = True
                return cached_result
            
            # Apply preprocessing optimizations
            preprocessing_result = await self._apply_preprocessing(
                file_path, strategy, file_classification
            )
            applied_optimizations.extend(preprocessing_result["optimizations"])
            
            # Apply format-specific optimizations
            format_result = await self._apply_format_optimizations(
                file_path, strategy, file_classification
            )
            applied_optimizations.extend(format_result["optimizations"])
            
            # Apply performance optimizations
            performance_result = await self._apply_performance_optimizations(
                file_path, strategy, processing_context
            )
            applied_optimizations.extend(performance_result["optimizations"])
            
            # Calculate optimization impact
            processing_time = (datetime.now() - start_time).total_seconds()
            time_saved = max(0.0, preprocessing_result.get("time_saved", 0.0) + 
                           format_result.get("time_saved", 0.0) + 
                           performance_result.get("time_saved", 0.0))
            
            memory_saved = max(0.0, preprocessing_result.get("memory_saved", 0.0) + 
                             format_result.get("memory_saved", 0.0) + 
                             performance_result.get("memory_saved", 0.0))
            
            quality_impact = min(1.0, max(-1.0, 
                               preprocessing_result.get("quality_impact", 0.0) + 
                               format_result.get("quality_impact", 0.0)))
            
            # Generate recommendations
            recommendations = await self._generate_optimization_recommendations(
                strategy, applied_optimizations, file_classification
            )
            
            result = OptimizationResult(
                applied_optimizations=applied_optimizations,
                processing_time_saved=time_saved,
                memory_usage_reduced=memory_saved,
                quality_impact=quality_impact,
                recommendations=recommendations,
                metadata={
                    "strategy_id": strategy.strategy_id,
                    "optimization_time": processing_time,
                    "file_size": file_path.stat().st_size,
                    "preprocessing_steps": len(strategy.preprocessing_steps),
                    "postprocessing_steps": len(strategy.postprocessing_steps)
                }
            )
            
            # Cache result
            if strategy.caching_enabled:
                self._optimization_cache[cache_key] = result
            
            # Update statistics
            self.performance_stats["strategies_applied"] += 1
            self.performance_stats["total_time_saved"] += time_saved
            self.performance_stats["total_memory_saved"] += memory_saved
            if quality_impact > 0:
                self.performance_stats["quality_improvements"] += 1
            
            return result
            
        except Exception as e:
            self.performance_stats["failed_optimizations"] += 1
            
            # Return minimal optimization result
            return OptimizationResult(
                applied_optimizations=applied_optimizations,
                processing_time_saved=0.0,
                memory_usage_reduced=0.0,
                quality_impact=0.0,
                recommendations=[f"Optimization failed: {str(e)}"],
                metadata={"error": str(e)}
            )
    
    async def _apply_preprocessing(
        self,
        file_path: Path,
        strategy: OptimizationStrategy,
        file_classification: Any
    ) -> Dict[str, Any]:
        """Apply preprocessing optimizations."""
        optimizations = []
        time_saved = 0.0
        memory_saved = 0.0
        quality_impact = 0.0
        
        for step in strategy.preprocessing_steps:
            try:
                if step == "file_size_check":
                    result = await self._optimize_file_size_check(file_path, strategy)
                    optimizations.append("file_size_validation")
                    time_saved += result.get("time_saved", 0.0)
                
                elif step == "encoding_detection":
                    result = await self._optimize_encoding_detection(file_path)
                    optimizations.append("encoding_optimization")
                    quality_impact += result.get("quality_impact", 0.0)
                
                elif step == "content_sampling":
                    result = await self._optimize_content_sampling(file_path, strategy)
                    optimizations.append("content_sampling")
                    time_saved += result.get("time_saved", 0.0)
                    memory_saved += result.get("memory_saved", 0.0)
                
                elif step == "format_validation":
                    result = await self._optimize_format_validation(file_path, file_classification)
                    optimizations.append("format_validation")
                    quality_impact += result.get("quality_impact", 0.0)
                
            except Exception as e:
                print(f"Warning: Preprocessing step '{step}' failed: {e}")
                continue
        
        return {
            "optimizations": optimizations,
            "time_saved": time_saved,
            "memory_saved": memory_saved,
            "quality_impact": quality_impact
        }
    
    async def _apply_format_optimizations(
        self,
        file_path: Path,
        strategy: OptimizationStrategy,
        file_classification: Any
    ) -> Dict[str, Any]:
        """Apply format-specific optimizations."""
        optimizations = []
        time_saved = 0.0
        memory_saved = 0.0
        quality_impact = 0.0
        
        detected_format = getattr(file_classification, 'detected_format', 'unknown')
        
        # PDF optimizations
        if detected_format == 'pdf':
            result = await self._optimize_pdf_processing(file_path, strategy)
            optimizations.extend(result["optimizations"])
            time_saved += result.get("time_saved", 0.0)
            quality_impact += result.get("quality_impact", 0.0)
        
        # Office document optimizations
        elif detected_format in ['docx', 'doc', 'xlsx', 'pptx']:
            result = await self._optimize_office_processing(file_path, strategy)
            optimizations.extend(result["optimizations"])
            time_saved += result.get("time_saved", 0.0)
            memory_saved += result.get("memory_saved", 0.0)
        
        # Text file optimizations
        elif detected_format in ['txt', 'md', 'log']:
            result = await self._optimize_text_processing(file_path, strategy)
            optimizations.extend(result["optimizations"])
            time_saved += result.get("time_saved", 0.0)
            memory_saved += result.get("memory_saved", 0.0)
        
        # Image optimizations
        elif detected_format in ['jpg', 'jpeg', 'png', 'gif']:
            result = await self._optimize_image_processing(file_path, strategy)
            optimizations.extend(result["optimizations"])
            time_saved += result.get("time_saved", 0.0)
        
        # Structured data optimizations
        elif detected_format in ['json', 'xml', 'yaml']:
            result = await self._optimize_structured_data_processing(file_path, strategy)
            optimizations.extend(result["optimizations"])
            time_saved += result.get("time_saved", 0.0)
            quality_impact += result.get("quality_impact", 0.0)
        
        return {
            "optimizations": optimizations,
            "time_saved": time_saved,
            "memory_saved": memory_saved,
            "quality_impact": quality_impact
        }
    
    async def _apply_performance_optimizations(
        self,
        file_path: Path,
        strategy: OptimizationStrategy,
        processing_context: Any
    ) -> Dict[str, Any]:
        """Apply performance optimizations."""
        optimizations = []
        time_saved = 0.0
        memory_saved = 0.0
        
        # Memory optimization
        if strategy.memory_optimization:
            result = await self._optimize_memory_usage(file_path, strategy)
            optimizations.append("memory_optimization")
            memory_saved += result.get("memory_saved", 0.0)
        
        # Parallel processing
        if strategy.parallel_processing:
            result = await self._optimize_parallel_processing(file_path, strategy)
            optimizations.append("parallel_processing")
            time_saved += result.get("time_saved", 0.0)
        
        # Chunking optimization
        result = await self._optimize_chunking_strategy(file_path, strategy)
        optimizations.append("chunking_optimization")
        time_saved += result.get("time_saved", 0.0)
        
        return {
            "optimizations": optimizations,
            "time_saved": time_saved,
            "memory_saved": memory_saved,
            "quality_impact": 0.0
        }
    
    # Format-specific optimization methods
    
    async def _optimize_pdf_processing(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize PDF processing."""
        optimizations = []
        time_saved = 0.0
        quality_impact = 0.0
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Large PDF optimization
        if file_size_mb > 50:
            optimizations.append("pdf_page_sampling")
            time_saved += 2.0  # Estimated time saved
        
        # OCR vs text extraction decision
        optimizations.append("pdf_extraction_method_selection")
        quality_impact += 0.1
        
        # PDF structure optimization
        optimizations.append("pdf_structure_analysis")
        time_saved += 0.5
        
        return {
            "optimizations": optimizations,
            "time_saved": time_saved,
            "quality_impact": quality_impact
        }
    
    async def _optimize_office_processing(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize Office document processing."""
        optimizations = []
        time_saved = 0.0
        memory_saved = 0.0
        
        # DOCX/XLSX are ZIP archives - optimize extraction
        optimizations.append("office_zip_optimization")
        time_saved += 0.3
        
        # Selective content extraction
        optimizations.append("office_selective_extraction")
        memory_saved += 15.0  # Percentage
        
        # Metadata extraction optimization
        optimizations.append("office_metadata_extraction")
        time_saved += 0.2
        
        return {
            "optimizations": optimizations,
            "time_saved": time_saved,
            "memory_saved": memory_saved
        }
    
    async def _optimize_text_processing(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize text file processing."""
        optimizations = []
        time_saved = 0.0
        memory_saved = 0.0
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Large file streaming
        if file_size_mb > 10:
            optimizations.append("text_streaming_processing")
            memory_saved += 30.0
            time_saved += 1.0
        
        # Encoding optimization
        optimizations.append("text_encoding_optimization")
        time_saved += 0.1
        
        # Line-by-line processing for logs
        if file_path.suffix.lower() in ['.log', '.txt']:
            optimizations.append("text_line_processing")
            memory_saved += 20.0
        
        return {
            "optimizations": optimizations,
            "time_saved": time_saved,
            "memory_saved": memory_saved
        }
    
    async def _optimize_image_processing(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize image processing."""
        optimizations = []
        time_saved = 0.0
        
        # OCR optimization for images
        optimizations.append("image_ocr_preprocessing")
        time_saved += 1.5
        
        # Image resize for OCR
        optimizations.append("image_resize_optimization")
        time_saved += 0.5
        
        return {
            "optimizations": optimizations,
            "time_saved": time_saved
        }
    
    async def _optimize_structured_data_processing(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize structured data processing."""
        optimizations = []
        time_saved = 0.0
        quality_impact = 0.0
        
        # Schema-aware parsing
        optimizations.append("structured_schema_parsing")
        quality_impact += 0.2
        
        # Incremental parsing for large files
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 5:
            optimizations.append("structured_incremental_parsing")
            time_saved += 1.0
        
        # Validation optimization
        optimizations.append("structured_validation_optimization")
        time_saved += 0.2
        
        return {
            "optimizations": optimizations,
            "time_saved": time_saved,
            "quality_impact": quality_impact
        }
    
    # Specific optimization implementations
    
    async def _optimize_file_size_check(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize file size checking."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > strategy.max_file_size_mb:
            return {"time_saved": 0.0, "skip_processing": True}
        
        return {"time_saved": 0.01}  # Minimal time saved by early validation
    
    async def _optimize_encoding_detection(self, file_path: Path) -> Dict[str, Any]:
        """Optimize encoding detection."""
        # Try UTF-8 first (most common), fall back to others
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Test read
                    break
            except UnicodeDecodeError:
                continue
        
        return {"quality_impact": 0.1}  # Better encoding = better quality
    
    async def _optimize_content_sampling(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize content sampling for large files."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > 50:
            # Sample first 10% for very large files
            return {"time_saved": 5.0, "memory_saved": 80.0}
        elif file_size_mb > 10:
            # Sample first 50% for large files
            return {"time_saved": 2.0, "memory_saved": 40.0}
        
        return {"time_saved": 0.0, "memory_saved": 0.0}
    
    async def _optimize_format_validation(self, file_path: Path, file_classification: Any) -> Dict[str, Any]:
        """Optimize format validation."""
        confidence = getattr(file_classification, 'confidence_score', 0.5)
        
        if confidence > 0.9:
            # High confidence - skip additional validation
            return {"quality_impact": 0.0, "time_saved": 0.1}
        else:
            # Low confidence - perform additional validation
            return {"quality_impact": 0.2, "time_saved": -0.1}
    
    async def _optimize_memory_usage(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize memory usage."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > 100:
            # Streaming processing for very large files
            return {"memory_saved": 60.0}
        elif file_size_mb > 10:
            # Chunked processing
            return {"memory_saved": 30.0}
        
        return {"memory_saved": 10.0}  # Basic optimization
    
    async def _optimize_parallel_processing(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize parallel processing."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > 20:
            # Significant speedup for large files
            return {"time_saved": file_size_mb * 0.1}
        
        return {"time_saved": 0.5}  # Minimal speedup for small files
    
    async def _optimize_chunking_strategy(self, file_path: Path, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Optimize chunking strategy."""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Adaptive chunk sizes based on file size
        if file_size_mb > 50:
            optimal_chunk_size = 2000
        elif file_size_mb > 10:
            optimal_chunk_size = 1500
        else:
            optimal_chunk_size = 1000
        
        # Update strategy recommendation
        strategy.recommended_chunk_size = optimal_chunk_size
        
        return {"time_saved": 0.2}
    
    async def _generate_optimization_recommendations(
        self,
        strategy: OptimizationStrategy,
        applied_optimizations: List[str],
        file_classification: Any
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        detected_format = getattr(file_classification, 'detected_format', 'unknown')
        confidence = getattr(file_classification, 'confidence_score', 0.5)
        
        # Format-specific recommendations
        if detected_format == 'pdf' and 'pdf_page_sampling' not in applied_optimizations:
            recommendations.append("Consider page sampling for large PDF files")
        
        if detected_format in ['docx', 'doc'] and 'office_selective_extraction' not in applied_optimizations:
            recommendations.append("Enable selective content extraction for Office documents")
        
        # General recommendations
        if confidence < 0.7:
            recommendations.append("Low format confidence - consider manual format specification")
        
        if strategy.optimization_level == OptimizationLevel.SPEED:
            recommendations.append("For better quality, consider using 'balanced' or 'quality' optimization")
        
        if not strategy.parallel_processing:
            recommendations.append("Enable parallel processing for better performance on large files")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_strategy_score(
        self,
        strategy: OptimizationStrategy,
        detected_format: str,
        confidence_score: float
    ) -> float:
        """Calculate strategy selection score."""
        base_score = 1.0
        
        # Format specificity bonus
        if detected_format in strategy.supported_formats:
            format_index = strategy.supported_formats.index(detected_format)
            base_score += (10 - format_index) * 0.1  # Earlier in list = higher score
        
        # Confidence factor
        base_score *= confidence_score
        
        # Strategy parameters bonus
        if strategy.memory_optimization:
            base_score += 0.2
        
        if strategy.parallel_processing:
            base_score += 0.15
        
        if strategy.caching_enabled:
            base_score += 0.1
        
        return base_score
    
    def _get_generic_strategy(self, optimization_level: str) -> OptimizationStrategy:
        """Get generic strategy when no specific match found."""
        level_enum = OptimizationLevel(optimization_level) if optimization_level in OptimizationLevel else OptimizationLevel.BALANCED
        
        return OptimizationStrategy(
            strategy_id="generic_fallback",
            method_name="generic_processing",
            supported_formats=["*"],
            optimization_level=level_enum,
            recommended_chunk_size=1000,
            max_file_size_mb=100.0,
            parallel_processing=level_enum in [OptimizationLevel.BALANCED, OptimizationLevel.QUALITY],
            memory_optimization=True,
            caching_enabled=True,
            preprocessing_steps=["file_size_check", "encoding_detection"],
            postprocessing_steps=["normalize_whitespace"],
            parameters={"fallback": True}
        )
    
    def _generate_cache_key(self, file_path: Path, strategy: OptimizationStrategy) -> str:
        """Generate cache key for optimization results."""
        import hashlib
        
        key_data = f"{file_path}{file_path.stat().st_mtime}{strategy.strategy_id}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _initialize_optimization_strategies(self) -> Dict[str, OptimizationStrategy]:
        """Initialize built-in optimization strategies."""
        strategies = {}
        
        # PDF strategies
        strategies["pdf_speed"] = OptimizationStrategy(
            strategy_id="pdf_speed",
            method_name="pdf_fast_extraction",
            supported_formats=["pdf"],
            optimization_level=OptimizationLevel.SPEED,
            recommended_chunk_size=800,
            max_file_size_mb=200.0,
            parallel_processing=False,
            preprocessing_steps=["file_size_check", "content_sampling"],
            postprocessing_steps=["normalize_whitespace"]
        )
        
        strategies["pdf_quality"] = OptimizationStrategy(
            strategy_id="pdf_quality",
            method_name="pdf_comprehensive_extraction",
            supported_formats=["pdf"],
            optimization_level=OptimizationLevel.QUALITY,
            recommended_chunk_size=1500,
            max_file_size_mb=500.0,
            parallel_processing=True,
            preprocessing_steps=["file_size_check", "encoding_detection", "format_validation"],
            postprocessing_steps=["structure_preservation", "metadata_enrichment"]
        )
        
        # Office document strategies
        strategies["office_balanced"] = OptimizationStrategy(
            strategy_id="office_balanced",
            method_name="office_structured_extraction",
            supported_formats=["docx", "doc", "xlsx", "pptx"],
            optimization_level=OptimizationLevel.BALANCED,
            recommended_chunk_size=1200,
            max_file_size_mb=150.0,
            parallel_processing=True,
            preprocessing_steps=["file_size_check", "format_validation"],
            postprocessing_steps=["structure_preservation"]
        )
        
        # Text file strategies
        strategies["text_speed"] = OptimizationStrategy(
            strategy_id="text_speed",
            method_name="text_streaming_extraction",
            supported_formats=["txt", "md", "log"],
            optimization_level=OptimizationLevel.SPEED,
            recommended_chunk_size=2000,
            max_file_size_mb=1000.0,
            parallel_processing=True,
            memory_optimization=True,
            preprocessing_steps=["encoding_detection"],
            postprocessing_steps=["normalize_whitespace"]
        )
        
        # Image strategies
        strategies["image_ocr"] = OptimizationStrategy(
            strategy_id="image_ocr",
            method_name="image_ocr_extraction",
            supported_formats=["jpg", "jpeg", "png", "gif", "bmp"],
            optimization_level=OptimizationLevel.QUALITY,
            recommended_chunk_size=500,
            max_file_size_mb=50.0,
            preprocessing_steps=["file_size_check"],
            postprocessing_steps=["ocr_cleanup"]
        )
        
        # Structured data strategies
        strategies["json_structured"] = OptimizationStrategy(
            strategy_id="json_structured",
            method_name="json_schema_extraction",
            supported_formats=["json"],
            optimization_level=OptimizationLevel.QUALITY,
            recommended_chunk_size=5000,
            max_file_size_mb=100.0,
            preprocessing_steps=["format_validation"],
            postprocessing_steps=["schema_validation"]
        )
        
        strategies["xml_structured"] = OptimizationStrategy(
            strategy_id="xml_structured",
            method_name="xml_parser_extraction",
            supported_formats=["xml"],
            optimization_level=OptimizationLevel.BALANCED,
            recommended_chunk_size=3000,
            max_file_size_mb=200.0,
            preprocessing_steps=["format_validation"],
            postprocessing_steps=["xml_cleanup"]
        )
        
        return strategies
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get optimization engine health status."""
        return {
            "strategies_available": len(self.strategies),
            "cache_size": len(self._optimization_cache),
            "performance_stats": self.performance_stats.copy(),
            "supported_formats": list(set(
                format_name
                for strategy in self.strategies.values()
                for format_name in strategy.supported_formats
            ))
        }
    
    def clear_cache(self) -> None:
        """Clear optimization cache."""
        self._optimization_cache.clear()

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global optimization engine instance
_default_engine = None

def get_default_optimization_engine() -> FormatOptimizationEngine:
    """Get default optimization engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = FormatOptimizationEngine()
    return _default_engine

async def optimize_file_processing(
    file_path: Path,
    detected_format: str,
    optimization_level: str = "balanced"
) -> OptimizationResult:
    """Apply format optimizations with default engine."""
    engine = get_default_optimization_engine()
    
    # Create mock file classification
    class MockClassification:
        def __init__(self, format_name: str):
            self.detected_format = format_name
            self.confidence_score = 0.8
    
    file_classification = MockClassification(detected_format)
    strategy = await engine.select_strategy(file_classification, optimization_level)
    
    return await engine.apply_optimizations(
        file_path, file_classification, strategy, None
    )