"""
API route handlers for the AI Text Detection REST API.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from api.models import (
    AnalyzeTextRequest,
    BatchAnalyzeRequest,
    AnalysisResponse,
    BatchAnalysisResponse,
    HealthResponse,
    InfoResponse,
)
from api.auth import verify_api_key
from api.utils import extract_text_from_file, validate_file_size
import config


# Create API router
router = APIRouter(prefix="/api/v1", tags=["AI Detection"])


# Global detector instance (set by main app during lifespan)
_detector = None


def get_detector():
    """Get the global detector instance."""
    if _detector is None:
        raise HTTPException(
            status_code=503,
            detail="Detector not initialized. Please wait for the server to fully start."
        )
    return _detector


def set_detector(detector):
    """Set the global detector instance (called during app startup)."""
    global _detector
    _detector = detector


def process_analysis_result(
    result: dict[str, Any],
    include_sentences: bool,
    include_metadata: bool
) -> dict[str, Any]:
    """
    Process detector result and filter based on request options.

    Args:
        result: Raw result from detector.analyze()
        include_sentences: Whether to include sentence-level analysis
        include_metadata: Whether to include metadata

    Returns:
        Filtered result dictionary
    """
    if not include_sentences:
        result.pop("sentences", None)

    if not include_metadata:
        result.pop("metadata", None)

    return result


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: AnalyzeTextRequest,
    _: str = Depends(verify_api_key)
) -> dict[str, Any]:
    """
    Analyze a single text for AI-generated content.

    **Authentication**: Requires valid API key in `X-API-Key` header.

    **Parameters**:
    - `text`: The text to analyze (1-50,000 characters)
    - `include_sentences`: Include per-sentence analysis (default: True)
    - `include_metadata`: Include document metadata (default: True)

    **Returns**:
    - `verdict`: Overall classification (AI, Human, Mixed, Unknown)
    - `ai_probability`: Probability score (0.0-1.0)
    - `confidence`: Confidence level (high, moderate, low)
    - `components`: Breakdown of detection signals
    - `sentences`: Per-sentence analysis (optional)
    - `metadata`: Document statistics (optional)
    """
    try:
        detector = get_detector()
        result = detector.analyze(request.text)
        return process_analysis_result(
            result,
            request.include_sentences,
            request.include_metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalyzeRequest,
    _: str = Depends(verify_api_key)
) -> dict[str, Any]:
    """
    Analyze multiple texts in parallel.

    **Authentication**: Requires valid API key in `X-API-Key` header.

    **Parameters**:
    - `texts`: List of texts to analyze (1-10 items, each max 50,000 chars)
    - `include_sentences`: Include per-sentence analysis (default: False)
    - `include_metadata`: Include metadata for each text (default: False)

    **Returns**:
    - `results`: Array of analysis results
    - `total_processed`: Number of texts processed
    - `processing_time_seconds`: Total processing time

    **Notes**:
    - Processes texts in parallel for better performance
    - Set `include_sentences=False` for faster batch processing
    """
    start_time = time.time()

    try:
        detector = get_detector()

        # Process texts in parallel using ThreadPoolExecutor
        def analyze_single(text: str) -> dict[str, Any]:
            result = detector.analyze(text)
            return process_analysis_result(
                result,
                request.include_sentences,
                request.include_metadata
            )

        with ThreadPoolExecutor(max_workers=min(len(request.texts), 4)) as executor:
            results = list(executor.map(analyze_single, request.texts))

        processing_time = time.time() - start_time

        return {
            "results": results,
            "total_processed": len(results),
            "processing_time_seconds": round(processing_time, 3)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...),
    include_sentences: bool = True,
    include_metadata: bool = True,
    _: str = Depends(verify_api_key)
) -> dict[str, Any]:
    """
    Upload and analyze a document file.

    **Authentication**: Requires valid API key in `X-API-Key` header.

    **Supported Formats**:
    - `.txt` - Plain text files
    - `.pdf` - PDF documents
    - `.docx` - Microsoft Word documents

    **Parameters**:
    - `file`: File to upload (max 10MB)
    - `include_sentences`: Include per-sentence analysis (query param, default: True)
    - `include_metadata`: Include document metadata (query param, default: True)

    **Returns**:
    - Same as `/analyze` endpoint

    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/analyze/file" \\
      -H "X-API-Key: your-key" \\
      -F "file=@document.pdf"
    ```
    """
    # Validate file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer

    validate_file_size(file_size, max_size_mb=config.API_MAX_FILE_SIZE_MB)

    # Extract text from file
    try:
        text = await extract_text_from_file(file)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process file: {str(e)}"
        )

    # Validate extracted text
    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text content found in file"
        )

    if len(text) > 50000:
        raise HTTPException(
            status_code=400,
            detail=f"Extracted text ({len(text)} chars) exceeds maximum length of 50,000 characters"
        )

    # Analyze the extracted text
    try:
        detector = get_detector()
        result = detector.analyze(text)
        return process_analysis_result(
            result,
            include_sentences,
            include_metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    **Authentication**: Not required.

    **Returns**:
    - `status`: Service status (healthy/unhealthy)
    - `model_loaded`: Whether detector is initialized
    - `classifier_trained`: Whether classifier is fine-tuned
    - `version`: API version

    **Use Cases**:
    - Load balancer health checks
    - Service monitoring
    - Startup verification
    """
    try:
        detector = get_detector()
        return {
            "status": "healthy",
            "model_loaded": True,
            "classifier_trained": detector.classifier.is_fine_tuned,
            "version": "1.0.0"
        }
    except Exception:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "classifier_trained": False,
            "version": "1.0.0"
        }


@router.get("/info", response_model=InfoResponse)
async def api_info() -> dict[str, Any]:
    """
    Get API and model information.

    **Authentication**: Not required.

    **Returns**:
    - `version`: API version
    - `model_status`: Loaded models and their training status
    - `ensemble_weights`: Current ensemble weights
    - `thresholds`: Detection thresholds used
    - `supported_models`: AI models that can be detected
    """
    try:
        detector = get_detector()

        return {
            "version": "1.0.0",
            "model_status": {
                "perplexity_model": config.PERPLEXITY_MODEL,
                "classifier_model": config.CLASSIFIER_MODEL,
                "classifier_trained": detector.classifier.is_fine_tuned,
            },
            "ensemble_weights": dict(detector.weights),
            "thresholds": {
                "perplexity_ai_threshold": config.PERPLEXITY_THRESHOLD_AI,
                "burstiness_ai_threshold": config.BURSTINESS_THRESHOLD_AI,
                "verdict_ai_threshold": config.VERDICT_AI_THRESHOLD,
                "verdict_human_threshold": config.VERDICT_HUMAN_THRESHOLD,
            },
            "supported_models": [
                "ChatGPT",
                "GPT-4",
                "Gemini",
                "Claude",
                "Llama",
                "Mistral",
                "DeepSeek"
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )
