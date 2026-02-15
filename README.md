# AI Text Detector

GPTZero-style AI text detection using **Perplexity**, **Burstiness**, and **DeBERTa** classification.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the web app (uses base models, no training needed)
streamlit run app.py
```

## Training (Optional — improves accuracy)

```bash
# Step 1: Download and prepare datasets
python -m training.prepare_data

# Step 2: Fine-tune DeBERTa classifier
python -m training.train_classifier

# Step 3: Evaluate performance
python -m training.evaluate
```

## Project Structure

```
├── detector/              # Core detection engine
│   ├── preprocessor.py    # Text normalization, sentence splitting
│   ├── perplexity.py      # GPT-2 perplexity engine
│   ├── burstiness.py      # Sentence-level PPL variance
│   ├── classifier.py      # DeBERTa binary classifier
│   └── ensemble.py        # Weighted ensemble combiner
├── training/              # Training pipeline
│   ├── prepare_data.py    # Dataset download & prep (HC3, dmitva, RAID, AI-pile, GPT-wiki)
│   ├── train_classifier.py # DeBERTa fine-tuning
│   └── evaluate.py        # Metrics & evaluation
├── tests/                 # Unit tests
├── app.py                 # Streamlit web interface
├── config.py              # All thresholds and settings
└── requirements.txt
```

## How It Works

1. **Perplexity** — GPT-2 measures how predictable text is. AI text has low perplexity.
2. **Burstiness** — Measures variance in sentence complexity. AI writes consistently; humans vary.
3. **DeBERTa Classifier** — Fine-tuned transformer for binary human/AI classification.
4. **Ensemble** — Weighted combination of all three signals → final verdict.
