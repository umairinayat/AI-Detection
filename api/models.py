"""
Pydantic models for request/response validation in the REST API.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator


class AnalyzeTextRequest(BaseModel):
    """Request model for single text analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Text to analyze (max 50,000 characters)"
    )
    include_sentences: bool = Field(
        default=True,
        description="Include per-sentence analysis in response"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata (language, document stats) in response"
    )

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch text analysis."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of texts to analyze (max 10 items)"
    )
    include_sentences: bool = Field(
        default=False,
        description="Include per-sentence analysis (affects performance)"
    )
    include_metadata: bool = Field(
        default=False,
        description="Include metadata for each text"
    )

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        for idx, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {idx} is empty")
            if len(text) > 50000:
                raise ValueError(f"Text at index {idx} exceeds 50,000 characters")
        return v


class ComponentScore(BaseModel):
    """Component-level scores (perplexity, burstiness, classifier)."""

    perplexity: Optional[dict[str, Any]] = None
    burstiness: Optional[dict[str, Any]] = None
    classifier: Optional[dict[str, Any]] = None


class SentenceAnalysis(BaseModel):
    """Per-sentence analysis details."""

    text: str
    perplexity: float
    ppl_ai_probability: float
    classifier_prob: float
    ai_probability: float


class AnalysisResponse(BaseModel):
    """Response model for text analysis."""

    verdict: str = Field(
        ...,
        description="Overall verdict: 'AI', 'Human', 'Mixed', or 'Unknown'"
    )
    ai_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that text is AI-generated (0-1)"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high', 'moderate', or 'low'"
    )
    confidence_category: str = Field(
        ...,
        description="Human-readable confidence description"
    )
    components: ComponentScore = Field(
        ...,
        description="Breakdown of component scores"
    )
    sentences: Optional[list[SentenceAnalysis]] = Field(
        default=None,
        description="Per-sentence analysis (if requested)"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Document metadata (if requested)"
    )


class BatchAnalysisResponse(BaseModel):
    """Response model for batch text analysis."""

    results: list[AnalysisResponse] = Field(
        ...,
        description="Analysis results for each text"
    )
    total_processed: int = Field(
        ...,
        description="Total number of texts processed"
    )
    processing_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in seconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    classifier_trained: bool = Field(default=False)
    version: str = Field(default="1.0.0")


class InfoResponse(BaseModel):
    """Response model for API info endpoint."""

    version: str
    model_status: dict[str, Any]
    ensemble_weights: dict[str, float]
    thresholds: dict[str, Any]
    supported_models: list[str]


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type/category")
    detail: str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
