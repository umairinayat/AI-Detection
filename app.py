"""
AI Text Detection — Streamlit Web Application

GPTZero-style interface with:
    - Text input area + file upload
    - Overall AI probability gauge
    - Confidence tier badges (High/Moderate/Low)
    - Per-sentence highlighting (green=human, red=AI)
    - Component-level breakdown
    - Perplexity distribution chart
    - Metadata insights
"""

import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector.ensemble import EnsembleDetector  # noqa: E402

# --- Page Config ---
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🔍",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .verdict-ai {
        background: linear-gradient(135deg, #ff4b4b, #ff6b6b);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .verdict-human {
        background: linear-gradient(135deg, #00c853, #69f0ae);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .verdict-mixed {
        background: linear-gradient(135deg, #ff9800, #ffb74d);
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .confidence-high {
        background: rgba(0, 200, 83, 0.2);
        color: #00c853;
        border: 1px solid #00c853;
    }
    .confidence-moderate {
        background: rgba(255, 152, 0, 0.2);
        color: #ff9800;
        border: 1px solid #ff9800;
    }
    .confidence-low {
        background: rgba(255, 75, 75, 0.2);
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
    }
    .sentence-ai {
        background-color: rgba(255, 75, 75, 0.15);
        border-left: 4px solid #ff4b4b;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 4px;
    }
    .sentence-human {
        background-color: rgba(0, 200, 83, 0.1);
        border-left: 4px solid #00c853;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 4px;
    }
    .sentence-uncertain {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 4px solid #ff9800;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .warning-banner {
        background: rgba(255, 152, 0, 0.15);
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .word-ai {
        background-color: rgba(255, 75, 75, var(--intensity, 0.3));
        border-radius: 3px;
        padding: 0 2px;
    }
    .word-human {
        background-color: rgba(0, 200, 83, var(--intensity, 0.3));
        border-radius: 3px;
        padding: 0 2px;
    }
    .token-indicators {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.3rem;
    }
    .token-ai-badge {
        background: rgba(255, 75, 75, 0.2);
        color: #ff6b6b;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    .token-human-badge {
        background: rgba(0, 200, 83, 0.15);
        color: #69f0ae;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load the ensemble detector (cached across reruns)."""
    # Prefer the newly trained llm_detector model
    llm_best_model = Path("models/detector/llm_detector/best")
    if llm_best_model.exists():
        return EnsembleDetector(classifier_path=str(llm_best_model))
    
    # Fallback to the older path
    best_model = Path("models/detector/best")
    if best_model.exists():
        return EnsembleDetector(classifier_path=str(best_model))
        
    return EnsembleDetector()


def render_gauge(probability: float):
    """Render a circular gauge chart showing AI probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48}},
        title={"text": "AI Probability", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#ff4b4b" if probability > 0.5 else "#00c853"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 30], "color": "rgba(0, 200, 83, 0.2)"},
                {"range": [30, 55], "color": "rgba(255, 152, 0, 0.15)"},
                {"range": [55, 75], "color": "rgba(255, 152, 0, 0.25)"},
                {"range": [75, 100], "color": "rgba(255, 75, 75, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def render_perplexity_chart(sentences: list[dict]):
    """Render a bar chart of per-sentence perplexity."""
    if not sentences:
        return None

    labels = [f"S{i+1}" for i in range(len(sentences))]
    ppls = [s["perplexity"] for s in sentences]
    colors = [
        "#ff4b4b" if s["ai_probability"] > 0.6
        else "#00c853" if s["ai_probability"] < 0.4
        else "#ff9800"
        for s in sentences
    ]

    fig = go.Figure(go.Bar(
        x=labels,
        y=ppls,
        marker_color=colors,
        text=[f"{p:.1f}" for p in ppls],
        textposition="auto",
    ))
    fig.update_layout(
        title="Sentence-Level Perplexity Distribution",
        xaxis_title="Sentence",
        yaxis_title="Perplexity",
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


def _highlight_words(text: str, word_attributions: list) -> str:
    """Build HTML with per-word color highlighting based on attribution scores."""
    if not word_attributions:
        return text

    # Build a mapping from word -> attribution score
    attr_map = {w.lower(): s for w, s in word_attributions}

    words = text.split()
    highlighted_parts = []
    for word in words:
        # Look up score for this word (case-insensitive, strip punctuation)
        clean = word.lower().strip(".,;:!?\"'()[]{}")
        score = attr_map.get(clean, 0.0)

        if abs(score) < 0.05:  # Below threshold — no highlight
            highlighted_parts.append(word)
        elif score > 0:  # AI indicator
            intensity = min(0.6, abs(score) * 0.6)
            highlighted_parts.append(
                f'<span class="word-ai" style="--intensity: {intensity:.2f}">{word}</span>'
            )
        else:  # Human indicator
            intensity = min(0.6, abs(score) * 0.6)
            highlighted_parts.append(
                f'<span class="word-human" style="--intensity: {intensity:.2f}">{word}</span>'
            )
    return " ".join(highlighted_parts)


def render_sentence_highlighting(sentences: list[dict]):
    """Render color-coded sentence-level analysis with word-level highlighting."""
    for i, s in enumerate(sentences):
        prob = s["ai_probability"]
        if prob > 0.6:
            css_class = "sentence-ai"
            label = f"AI ({prob:.0%})"
            icon = "🔴"
        elif prob < 0.4:
            css_class = "sentence-human"
            label = f"Human ({1-prob:.0%})"
            icon = "🟢"
        else:
            css_class = "sentence-uncertain"
            label = f"Uncertain ({prob:.0%})"
            icon = "🟡"

        # Build word-level highlighted text
        word_attrs = s.get("word_attributions", [])
        highlighted_text = _highlight_words(s["text"], word_attrs)

        # Build token indicator badges
        token_html = ""
        top_ai = s.get("top_ai_tokens", [])
        top_human = s.get("top_human_tokens", [])
        if top_ai or top_human:
            badges = []
            for w, score in top_ai[:3]:
                badges.append(f'<span class="token-ai-badge">🔴 {w} ({score:+.2f})</span>')
            for w, score in top_human[:3]:
                badges.append(f'<span class="token-human-badge">🟢 {w} ({score:+.2f})</span>')
            token_html = f'<div class="token-indicators">{" ".join(badges)}</div>'

        st.markdown(
            f'<div class="{css_class}">'
            f'<small><b>{icon} {label}</b> | PPL: {s["perplexity"]:.1f}</small><br/>'
            f'{highlighted_text}'
            f'{token_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_confidence_badge(confidence: str, category: str):
    """Render a confidence tier badge."""
    css_class = f"confidence-{confidence}"
    st.markdown(
        f'<div style="text-align: center;">'
        f'<span class="confidence-badge {css_class}">'
        f'{category}'
        f'</span></div>',
        unsafe_allow_html=True,
    )


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from uploaded file (TXT, PDF, DOCX)."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")

    elif filename.endswith(".pdf"):
        try:
            import pdfplumber
            import io
            pdf_bytes = io.BytesIO(uploaded_file.read())
            with pdfplumber.open(pdf_bytes) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
                return "\n\n".join(pages)
        except ImportError:
            st.error("PDF support requires `pdfplumber`. Install: `pip install pdfplumber`")
            return ""

    elif filename.endswith(".docx"):
        try:
            from docx import Document
            import io
            doc_bytes = io.BytesIO(uploaded_file.read())
            doc = Document(doc_bytes)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            st.error("DOCX support requires `python-docx`. Install: `pip install python-docx`")
            return ""

    else:
        st.error(f"Unsupported file type: {filename}")
        return ""


# --- Main App ---
def main():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🔍 AI Text Detector")
    st.caption("Detect AI-generated text using Perplexity, Burstiness, and Deep Learning")
    st.markdown('</div>', unsafe_allow_html=True)

    # Load detector
    with st.spinner("Loading detection models..."):
        detector = load_detector()

    # Sidebar
    with st.sidebar:
        st.header("How It Works")
        st.markdown("""
        **Three detection signals:**

        1. **Perplexity** -- How predictable is the text?
           AI text is highly predictable (low PPL).

        2. **Burstiness** -- How variable is sentence complexity?
           AI writes at a consistent level; humans vary.

        3. **DeBERTa Classifier** -- Neural network trained on
           labeled human/AI text pairs.

        The ensemble combines these signals with calibrated weights.
        """)

        st.divider()

        st.markdown("**Confidence Levels:**")
        st.markdown("""
        - **High**: Error rate < 1%
        - **Moderate**: Good confidence, verify with context
        - **Low**: Uncertain, use as a starting point only
        """)

        st.divider()

        st.markdown("**Model Status:**")
        if detector.classifier.is_fine_tuned:
            st.success("Fine-tuned classifier loaded")
            st.caption(f"Weights: PPL={detector.weights['perplexity']}, "
                      f"Burst={detector.weights['burstiness']}, "
                      f"CLF={detector.weights['classifier']}")
        else:
            st.warning("Using base model (not fine-tuned). "
                      "Detection relies on perplexity and burstiness only. "
                      "Run training for better accuracy.")
            st.caption("Classifier weight set to 0 to avoid noise.")

        st.divider()
        st.markdown("**Supported models:**")
        st.caption("ChatGPT, GPT-4, Gemini, Claude, Llama, Mistral, DeepSeek")

    # Input: text area or file upload
    tab1, tab2 = st.tabs(["Paste Text", "Upload File"])

    with tab1:
        text_input = st.text_area(
            "Paste text to analyze:",
            height=250,
            placeholder="Enter at least 2-3 sentences for accurate detection...",
            max_chars=50000,
        )
        char_count = len(text_input) if text_input else 0
        st.caption(f"{char_count:,}/50,000 characters")

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt", "pdf", "docx"],
            help="Supported formats: TXT, PDF, DOCX"
        )
        if uploaded_file:
            text_input = extract_text_from_upload(uploaded_file)
            if text_input:
                st.success(f"Extracted {len(text_input):,} characters from {uploaded_file.name}")
                with st.expander("Preview extracted text"):
                    st.text(text_input[:2000] + ("..." if len(text_input) > 2000 else ""))

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_btn = st.button("Analyze Text", use_container_width=True, type="primary")

    if analyze_btn and text_input and text_input.strip():
        with st.spinner("Analyzing text... (this may take a moment on CPU)"):
            result = detector.analyze(text_input)

        # --- Verdict Banner ---
        verdict = result["verdict"]
        prob = result["ai_probability"]
        confidence = result.get("confidence", "low")
        confidence_category = result.get("confidence_category", "")

        if verdict == "AI":
            st.markdown(
                f'<div class="verdict-ai">Verdict: AI-Generated ({prob:.0%})</div>',
                unsafe_allow_html=True,
            )
        elif verdict == "Human":
            st.markdown(
                f'<div class="verdict-human">Verdict: Human-Written ({1-prob:.0%})</div>',
                unsafe_allow_html=True,
            )
        elif verdict == "Mixed":
            st.markdown(
                f'<div class="verdict-mixed">Verdict: Mixed Authorship</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Could not determine authorship. Please provide more text.")

        # Confidence badge
        render_confidence_badge(confidence, confidence_category)

        st.write("")

        # --- Metadata warnings ---
        metadata = result.get("metadata", {})
        if metadata.get("homoglyphs_detected"):
            st.markdown(
                '<div class="warning-banner">'
                '⚠️ <b>Homoglyph characters detected.</b> '
                'This text contains Unicode lookalike characters that may be used '
                'to evade AI detection. Characters have been normalized for analysis.'
                '</div>',
                unsafe_allow_html=True,
            )
        if not metadata.get("classifier_trained", False):
            st.markdown(
                '<div class="warning-banner">'
                'Note: The DeBERTa classifier is not fine-tuned. '
                'Results are based on perplexity and burstiness analysis only. '
                'Train the classifier for higher accuracy.'
                '</div>',
                unsafe_allow_html=True,
            )

        # --- Component Breakdown ---
        st.subheader("Component Breakdown")
        comp = result["components"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Perplexity",
                f'{comp["perplexity"]["global_ppl"]:.1f}',
                help="Lower = more AI-like. AI text typically 7-15, human 25+."
            )
            ppl_prob = comp["perplexity"]["ai_probability"]
            bar_color = "🔴" if ppl_prob > 0.6 else "🟢" if ppl_prob < 0.4 else "🟡"
            st.caption(f'{bar_color} AI Prob: {ppl_prob:.0%}')

        with col2:
            st.metric(
                "Burstiness",
                f'{comp["burstiness"]["score"]:.3f}',
                help="Lower = more AI-like (flat writing pattern). Human writing is bursty."
            )
            burst_prob = comp["burstiness"]["ai_probability"]
            bar_color = "🔴" if burst_prob > 0.6 else "🟢" if burst_prob < 0.4 else "🟡"
            st.caption(f'{bar_color} AI Prob: {burst_prob:.0%}')

        with col3:
            clf_prob = comp["classifier"]["ai_probability"]
            is_ft = comp["classifier"]["is_fine_tuned"]
            st.metric(
                "Classifier",
                f'{clf_prob:.0%}',
                help="Neural network prediction. Most accurate when fine-tuned."
            )
            status = "Fine-tuned" if is_ft else "Not trained"
            bar_color = "🔴" if clf_prob > 0.6 else "🟢" if clf_prob < 0.4 else "🟡"
            st.caption(f'{bar_color} AI Prob: {clf_prob:.0%} ({status})')

        # --- Gauge ---
        st.subheader("Overall Score")
        gauge_fig = render_gauge(result["ai_probability"])
        st.plotly_chart(gauge_fig, use_container_width=True)

        # --- Perplexity chart ---
        if result["sentences"]:
            ppl_fig = render_perplexity_chart(result["sentences"])
            if ppl_fig:
                st.plotly_chart(ppl_fig, use_container_width=True)

        # --- Sentence highlighting ---
        st.subheader("Sentence-Level Analysis")
        st.caption("Each sentence is color-coded: 🟢 Human | 🟡 Uncertain | 🔴 AI. "
                   "Individual words are highlighted by their attribution score.")
        if result["sentences"]:
            render_sentence_highlighting(result["sentences"])
        else:
            st.info("No sentences to analyze. Provide longer text for detailed analysis.")

        # --- Document info ---
        if metadata:
            with st.expander("Document Info"):
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.metric("Characters", f'{metadata.get("original_length", 0):,}')
                with info_cols[1]:
                    st.metric("Sentences", metadata.get("num_sentences", 0))
                with info_cols[2]:
                    st.metric("Language", metadata.get("language", "unknown").upper())
                with info_cols[3]:
                    st.metric("Classifier", "Trained" if metadata.get("classifier_trained") else "Base")

        # --- Raw JSON ---
        with st.expander("Raw Detection Data"):
            st.json(result)

    elif analyze_btn:
        st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
