"""
AI Text Detection REST API

FastAPI-based REST API for AI-generated text detection.
Runs independently from the Streamlit app on port 8000.

Features:
- Single text analysis
- Batch text analysis (up to 10 texts)
- File upload analysis (TXT, PDF, DOCX)
- API key authentication
- Auto-generated OpenAPI documentation at /docs

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from detector.ensemble import EnsembleDetector
from api.endpoints import router, set_detector
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads the detector model on startup and cleans up on shutdown.
    """
    # Startup: Load detector model
    print("=" * 60)
    print("🚀 Starting AI Text Detection API...")
    print("=" * 60)

    try:
        # Check for fine-tuned model
        best_model = Path("models/detector/best")
        if best_model.exists():
            print(f"✓ Loading fine-tuned classifier from {best_model}")
            detector = EnsembleDetector(classifier_path=str(best_model))
        else:
            print("⚠ No fine-tuned model found, using base models")
            detector = EnsembleDetector()

        # Set global detector instance
        set_detector(detector)

        print("✓ Detector loaded successfully")
        print(f"  - Perplexity model: {config.PERPLEXITY_MODEL}")
        print(f"  - Classifier model: {config.CLASSIFIER_MODEL}")
        print(f"  - Classifier trained: {detector.classifier.is_fine_tuned}")
        print(f"  - Ensemble weights: {detector.weights}")
        print("=" * 60)
        print("✓ API ready to accept requests")
        print("📖 Documentation: http://localhost:8000/docs")
        print("🏥 Health check: http://localhost:8000/api/v1/health")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Failed to load detector: {e}")
        raise

    yield

    # Shutdown: Cleanup (if needed)
    print("\n" + "=" * 60)
    print("🛑 Shutting down API server...")
    print("=" * 60)


# Create FastAPI app
app = FastAPI(
    title="AI Text Detection API",
    description="""
    REST API for detecting AI-generated text using multi-signal analysis.

    ## Features

    - **Single Text Analysis**: Analyze individual texts up to 50,000 characters
    - **Batch Processing**: Analyze up to 10 texts in parallel
    - **File Upload**: Support for TXT, PDF, and DOCX files
    - **Multi-Signal Detection**: Combines perplexity, burstiness, and deep learning
    - **Confidence Scoring**: High/Moderate/Low confidence levels

    ## Authentication

    All analysis endpoints require an API key passed via the `X-API-Key` header.

    ## Detection Method

    The API uses a GPTZero-inspired ensemble approach:
    1. **Perplexity** - Measures text predictability using GPT-2
    2. **Burstiness** - Analyzes sentence-level variance
    3. **DeBERTa Classifier** - Fine-tuned transformer for binary classification

    ## Supported AI Models

    Detects text from: ChatGPT, GPT-4, Gemini, Claude, Llama, Mistral, DeepSeek
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - allow requests from any origin (configure as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with detailed messages."""
    errors = exc.errors()
    error_messages = []

    for error in errors:
        field = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        error_messages.append(f"{field}: {message}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "detail": "; ".join(error_messages),
            "status_code": 422
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": str(exc),
            "status_code": 500
        }
    )


# Include API routes
app.include_router(router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    API root endpoint with basic information and links.
    """
    return {
        "message": "AI Text Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "api_info": "/api/v1/info",
        "endpoints": {
            "analyze_text": "POST /api/v1/analyze",
            "analyze_batch": "POST /api/v1/analyze/batch",
            "analyze_file": "POST /api/v1/analyze/file",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info"
    )
