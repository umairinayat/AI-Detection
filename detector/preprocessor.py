"""
Text preprocessor for the AI detection pipeline.

Handles:
  - Text normalization
  - Homoglyph/Unicode attack defense
  - Sentence splitting
  - Language detection (basic)
  - Paragraph segmentation
"""

import re
import unicodedata
import nltk

# Download punkt tokenizer data (needed once)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


# --- Homoglyph normalization map ---
# Common Unicode lookalikes used to evade AI detectors
HOMOGLYPH_MAP = {
    # Cyrillic lookalikes
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E",
    "\u041d": "H", "\u041a": "K", "\u041c": "M", "\u041e": "O",
    "\u0420": "P", "\u0422": "T", "\u0425": "X",
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x",
    # Greek lookalikes
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N",
    "\u039f": "O", "\u03a1": "P", "\u03a4": "T", "\u03a7": "X",
    "\u03b1": "a", "\u03bf": "o",
    # Full-width Latin
    "\uff21": "A", "\uff22": "B", "\uff23": "C", "\uff24": "D",
    "\uff25": "E", "\uff26": "F", "\uff27": "G", "\uff28": "H",
    "\uff29": "I", "\uff2a": "J", "\uff2b": "K", "\uff2c": "L",
    "\uff2d": "M", "\uff2e": "N", "\uff2f": "O", "\uff30": "P",
    "\uff31": "Q", "\uff32": "R", "\uff33": "S", "\uff34": "T",
    "\uff35": "U", "\uff36": "V", "\uff37": "W", "\uff38": "X",
    "\uff39": "Y", "\uff3a": "Z",
}


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to defend against homoglyph attacks.

    1. Apply NFKC normalization (converts full-width, compatibility chars)
    2. Replace known homoglyphs with ASCII equivalents
    3. Strip zero-width and invisible characters
    """
    # Step 1: NFKC normalization (handles full-width, superscripts, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Replace known homoglyphs
    for homoglyph, replacement in HOMOGLYPH_MAP.items():
        text = text.replace(homoglyph, replacement)

    # Step 3: Remove zero-width and invisible characters
    # Comprehensive list of invisible/formatting Unicode chars
    invisible_pattern = re.compile(
        r"[\u200b\u200c\u200d\u200e\u200f"   # Zero-width chars, direction marks
        r"\u202a\u202b\u202c\u202d\u202e"      # Bidirectional formatting
        r"\u2060\u2061\u2062\u2063\u2064"       # Word joiner, invisible operators
        r"\u2066\u2067\u2068\u2069"              # Isolate marks
        r"\ufeff\ufff9\ufffa\ufffb"              # BOM, interlinear annotation
        r"\u00ad"                                 # Soft hyphen
        r"\u034f"                                 # Combining grapheme joiner
        r"\u061c"                                 # Arabic letter mark
        r"\u180e"                                 # Mongolian vowel separator
        r"]"
    )
    text = invisible_pattern.sub("", text)

    return text


def normalize_text(text: str) -> str:
    """Full text normalization: Unicode defense + whitespace cleanup."""
    # Unicode normalization first (handles homoglyphs, invisible chars)
    text = normalize_unicode(text)

    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out very short fragments."""
    sentences = sent_tokenize(text)
    # Keep only sentences with enough content to analyze
    # Lowered from 15 to 10 chars to keep short meaningful sentences
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs (for paragraph-level analysis).
    Preserves meaningful paragraph boundaries.
    """
    # Split on double newlines or multiple newlines
    paragraphs = re.split(r"\n\s*\n", text)
    # Filter empty paragraphs
    return [p.strip() for p in paragraphs if len(p.strip()) > 20]


def detect_language(text: str) -> str:
    """
    Basic language detection based on character analysis.
    Returns ISO 639-1 language code or 'unknown'.

    Note: For production, consider using langdetect or fasttext.
    """
    # Simple heuristic: check for non-Latin characters
    latin_count = sum(1 for c in text if c.isascii() and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return "unknown"

    latin_ratio = latin_count / total_alpha

    if latin_ratio > 0.8:
        return "en"  # Assume English for Latin-script text
    else:
        return "other"


def preprocess(text: str) -> dict:
    """
    Full preprocessing pipeline.

    Returns:
        dict with keys:
            - "full_text": normalized full text
            - "sentences": list of cleaned sentences
            - "paragraphs": list of paragraph strings
            - "language": detected language code
            - "original_length": character count of original text
            - "homoglyphs_detected": whether homoglyph substitution was found
    """
    original_length = len(text)

    # Detect language first to gate homoglyph detection
    language = detect_language(text)

    # Only flag homoglyphs in Latin-script text (e.g. English).
    # In Cyrillic/Greek text, these characters are legitimate and not
    # an evasion attack. Homoglyph attacks only make sense when
    # non-Latin characters are injected into primarily Latin text.
    if language == "en":
        homoglyphs_detected = any(c in text for c in HOMOGLYPH_MAP)
    else:
        homoglyphs_detected = False

    normalized = normalize_text(text)
    sentences = split_sentences(normalized)
    paragraphs = split_paragraphs(text)  # Use original text for paragraph splits

    return {
        "full_text": normalized,
        "sentences": sentences,
        "paragraphs": paragraphs,
        "language": language,
        "original_length": original_length,
        "homoglyphs_detected": homoglyphs_detected,
    }
