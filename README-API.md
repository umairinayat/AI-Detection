# AI Text Detection REST API

REST API for detecting AI-generated text using a multi-signal ensemble approach inspired by GPTZero.

## Features

- **Single Text Analysis** - Analyze individual texts (up to 50,000 characters)
- **Batch Processing** - Analyze up to 10 texts in parallel
- **File Upload** - Support for TXT, PDF, and DOCX files
- **API Key Authentication** - Secure access control
- **Auto-generated Documentation** - Interactive Swagger UI at `/docs`
- **Multi-Signal Detection** - Combines perplexity, burstiness, and deep learning

## Quick Start

### 1. Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install API-specific requirements
pip install -r requirements-api.txt
```

### 2. Configure API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set your API key
# The system will auto-generate one on first run if none exists
```

### 3. Start the API Server

```bash
# Development mode (with auto-reload)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
```

### 4. Access Documentation

Open your browser and visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## API Endpoints

### 🔍 Analyze Single Text

**POST** `/api/v1/analyze`

Analyze a single text for AI-generated content.

**Request Body:**
```json
{
  "text": "Your text to analyze here...",
  "include_sentences": true,
  "include_metadata": true
}
```

**Response:**
```json
{
  "verdict": "AI",
  "ai_probability": 0.87,
  "confidence": "high",
  "confidence_category": "Highly confident this is AI-generated",
  "components": {
    "perplexity": {
      "global_ppl": 12.5,
      "ai_probability": 0.92
    },
    "burstiness": {
      "score": 0.23,
      "ai_probability": 0.88
    },
    "classifier": {
      "ai_probability": 0.85,
      "is_fine_tuned": true
    }
  },
  "sentences": [...],
  "metadata": {...}
}
```

### 📦 Batch Analysis

**POST** `/api/v1/analyze/batch`

Analyze multiple texts in parallel (max 10).

**Request Body:**
```json
{
  "texts": [
    "First text to analyze...",
    "Second text to analyze...",
    "Third text to analyze..."
  ],
  "include_sentences": false,
  "include_metadata": false
}
```

**Response:**
```json
{
  "results": [
    { "verdict": "AI", "ai_probability": 0.87, ... },
    { "verdict": "Human", "ai_probability": 0.15, ... },
    { "verdict": "Mixed", "ai_probability": 0.52, ... }
  ],
  "total_processed": 3,
  "processing_time_seconds": 2.45
}
```

### 📄 File Upload Analysis

**POST** `/api/v1/analyze/file`

Upload and analyze a document file (TXT, PDF, DOCX).

**Form Data:**
- `file`: File to upload (max 10MB)
- `include_sentences`: boolean (query param, default: true)
- `include_metadata`: boolean (query param, default: true)

**Response:** Same as single text analysis

### 🏥 Health Check

**GET** `/api/v1/health`

Check API health status. No authentication required.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classifier_trained": true,
  "version": "1.0.0"
}
```

### ℹ️ API Info

**GET** `/api/v1/info`

Get API and model information. No authentication required.

**Response:**
```json
{
  "version": "1.0.0",
  "model_status": {
    "perplexity_model": "gpt2",
    "classifier_model": "microsoft/deberta-v3-base",
    "classifier_trained": true
  },
  "ensemble_weights": {
    "perplexity": 0.25,
    "burstiness": 0.20,
    "classifier": 0.55
  },
  "thresholds": {...},
  "supported_models": ["ChatGPT", "GPT-4", "Gemini", "Claude", ...]
}
```

## Authentication

All analysis endpoints require an API key passed in the `X-API-Key` header.

### Get Your API Key

On first run, the system automatically generates an API key and saves it to `.api_keys.json`. Check the console output for your key.

Alternatively, set it in `.env`:
```env
API_KEY_1=aidet_your-secret-key-here
```

### Using API Keys

```bash
# In curl
curl -H "X-API-Key: your-key-here" http://localhost:8000/api/v1/analyze

# In Python requests
import requests

headers = {"X-API-Key": "your-key-here"}
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={"text": "Your text here"},
    headers=headers
)
```

## Usage Examples

### cURL Examples

```bash
# Health check (no auth required)
curl http://localhost:8000/api/v1/health

# Analyze text
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample text to analyze for AI generation.",
    "include_sentences": true
  }'

# Batch analysis
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First text here",
      "Second text here",
      "Third text here"
    ]
  }'

# Upload file
curl -X POST http://localhost:8000/api/v1/analyze/file \
  -H "X-API-Key: your-key" \
  -F "file=@document.pdf"
```

### Python Examples

```python
import requests

API_URL = "http://localhost:8000/api/v1"
API_KEY = "your-key-here"
headers = {"X-API-Key": API_KEY}

# Analyze single text
response = requests.post(
    f"{API_URL}/analyze",
    json={
        "text": "Your text to analyze here...",
        "include_sentences": True
    },
    headers=headers
)
result = response.json()
print(f"Verdict: {result['verdict']}")
print(f"AI Probability: {result['ai_probability']:.2%}")

# Batch analysis
response = requests.post(
    f"{API_URL}/analyze/batch",
    json={
        "texts": [
            "First text",
            "Second text",
            "Third text"
        ],
        "include_sentences": False
    },
    headers=headers
)
results = response.json()
for i, result in enumerate(results['results']):
    print(f"Text {i+1}: {result['verdict']} ({result['ai_probability']:.2%})")

# Upload file
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{API_URL}/analyze/file",
        files={"file": f},
        headers=headers
    )
result = response.json()
print(f"File analysis: {result['verdict']}")
```

### JavaScript/Node.js Examples

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000/api/v1';
const API_KEY = 'your-key-here';

// Analyze text
async function analyzeText(text) {
  const response = await axios.post(
    `${API_URL}/analyze`,
    {
      text: text,
      include_sentences: true
    },
    {
      headers: { 'X-API-Key': API_KEY }
    }
  );
  return response.data;
}

// Usage
analyzeText("Your text here")
  .then(result => {
    console.log(`Verdict: ${result.verdict}`);
    console.log(`AI Probability: ${result.ai_probability}`);
  });
```

## Running with Streamlit App

You can run both the REST API and Streamlit app simultaneously:

```bash
# Terminal 1: Start REST API
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit UI
streamlit run app.py
```

- **REST API**: http://localhost:8000
- **Streamlit UI**: http://localhost:8501

## Configuration

### Environment Variables

Set in `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY_1` | Auto-generated | Primary API key |
| `API_HOST` | `0.0.0.0` | Server host |
| `API_PORT` | `8000` | Server port |
| `API_DEBUG` | `false` | Debug mode |

### API Limits

- **Max text length**: 50,000 characters
- **Max file size**: 10MB
- **Max batch size**: 10 texts
- **Supported file types**: TXT, PDF, DOCX

## Error Handling

All errors return consistent JSON format:

```json
{
  "error": "ValidationError",
  "detail": "Text exceeds maximum length of 50,000 characters",
  "status_code": 400
}
```

### HTTP Status Codes

- `200` - Success
- `400` - Bad request (invalid input)
- `401` - Unauthorized (missing/invalid API key)
- `413` - Payload too large (file size exceeded)
- `422` - Validation error
- `500` - Internal server error

## Detection Method

The API uses a **GPTZero-inspired ensemble approach**:

1. **Perplexity Analysis** (GPT-2)
   - Measures text predictability
   - AI text: low perplexity (7-15)
   - Human text: high perplexity (25+)

2. **Burstiness Analysis**
   - Measures sentence-level variance
   - AI: consistent (low burstiness)
   - Human: varying (high burstiness)

3. **DeBERTa Classifier**
   - Fine-tuned transformer
   - Binary classification (Human vs AI)

**Ensemble weights** (with trained classifier):
- Classifier: 55%
- Perplexity: 25%
- Burstiness: 20%

## Supported AI Models

Detects text from:
- ChatGPT (GPT-3.5, GPT-4)
- Google Gemini
- Anthropic Claude
- Meta Llama
- Mistral AI
- DeepSeek

## Performance Considerations

- **First request**: May be slower due to model loading
- **Concurrent requests**: Use multiple workers for production
- **GPU acceleration**: Automatically used if available
- **Batch processing**: More efficient than multiple single requests

## Troubleshooting

### API key not working
- Check `.api_keys.json` file exists
- Verify key format: `aidet_<random-string>`
- Ensure `X-API-Key` header is included

### Models not loading
- Check `models/detector/best` directory exists for fine-tuned model
- Verify internet connection for downloading base models
- Check disk space and memory

### File upload errors
- Ensure `pdfplumber` installed: `pip install pdfplumber`
- Ensure `python-docx` installed: `pip install python-docx`
- Check file size < 10MB
- Verify file format (TXT, PDF, DOCX only)

## Development

### Adding New API Keys

Edit `.api_keys.json`:
```json
[
  "aidet_key1_here",
  "aidet_key2_here"
]
```

Or use environment variables in `.env`:
```env
API_KEY_1=aidet_key1
API_KEY_2=aidet_key2
```

### Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run API tests
pytest tests/test_api.py
```

## Production Deployment

### Using Gunicorn (Linux/Mac)

```bash
pip install gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt -r requirements-api.txt

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### CORS Configuration

Edit `api.py` to restrict origins in production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## License

Same as the main AI Detection project.

## Support

For issues and questions:
- Check the interactive docs at `/docs`
- Review error messages in API responses
- Check server logs for detailed errors

---

**Built with FastAPI** | Powered by GPT-2, DeBERTa, and ensemble AI detection
