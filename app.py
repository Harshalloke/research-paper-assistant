import re

import streamlit as st
import pdfplumber

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import sent_tokenize

from wordfreq import zipf_frequency  # dictionary-based word frequency


# =========================
# NLTK SETUP (punkt + punkt_tab)
# =========================

def setup_nltk():
    # Punkt sentence tokenizer
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # New requirement in latest NLTK: punkt_tab (for newer versions)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

setup_nltk()


# =========================
# PDF TEXT EXTRACTION
# =========================

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from an uploaded PDF file (Streamlit's UploadedFile).
    """
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    # Normalize newlines a bit
    text = text.replace("\r", "\n")
    return text


# =========================
# SECTION SPLITTING
# =========================

SECTION_TITLES = [
    "abstract",
    "introduction",
    "literature review",
    "background",
    "related work",
    "methodology",
    "methods",
    "materials and methods",
    "experiments",
    "results",
    "discussion",
    "results and discussion",
    "analysis",
    "conclusion",
    "conclusions",
    "future work",
    "references",
]


def split_into_sections(text: str):
    """
    Roughly split a research paper into sections based on common headings.
    Returns dict: {section_name: section_text}
    """
    sections = {}
    if not text or len(text.strip()) == 0:
        return sections

    lower_text = text.lower()
    positions = []

    # Find all occurrences of section titles
    for title in SECTION_TITLES:
        # Look for the title as a standalone line (approx)
        pattern = r"\n\s*" + re.escape(title) + r"\s*\n"
        for match in re.finditer(pattern, lower_text):
            positions.append((match.start(), title))

    # If no headings found, return whole text as one section
    if not positions:
        sections["Full Paper"] = text.strip()
        return sections

    positions.sort(key=lambda x: x[0])

    # Build sections using positions
    for i, (start_idx, title) in enumerate(positions):
        end_idx = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        raw_section_text = text[start_idx:end_idx].strip()

        # Remove the heading line itself from section content
        first_newline = raw_section_text.find("\n")
        if first_newline != -1:
            section_body = raw_section_text[first_newline:].strip()
        else:
            section_body = raw_section_text

        pretty_title = title.title()
        if pretty_title in sections:
            # join if repeated (e.g. multiple "Results" segments)
            sections[pretty_title] += "\n" + section_body
        else:
            sections[pretty_title] = section_body

    return sections


# =========================
# HELPER: FIX DOUBLED-LETTER WORDS
# =========================

def _undouble_word_if_pattern(word: str) -> str:
    """
    Fix words like 'SSyyrraaccuussee' → 'Syracuse'
    Only if:
      - length is even and >= 6
      - every pair of characters is the same letter (case-insensitive)
    Keeps normal words like 'book', 'letter', 'processing' safe.
    """
    # Only operate on pure alpha words
    if not re.fullmatch(r"[A-Za-z]+", word):
        return word

    if len(word) < 6 or len(word) % 2 != 0:
        return word

    chars = list(word)
    pairs = [chars[i:i+2] for i in range(0, len(chars), 2)]

    for a, b in pairs:
        if a.lower() != b.lower():
            return word  # not the doubled pattern

    # All pairs are doubled – compress
    fixed = "".join(p[0] for p in pairs)
    # Nice capitalization
    return fixed[0].upper() + fixed[1:]


def _fix_doubled_words_in_text(text: str) -> str:
    """
    Apply _undouble_word_if_pattern to each token, preserving spaces.
    """
    tokens = re.split(r"(\s+)", text)  # keep separators
    fixed_tokens = []
    for tok in tokens:
        if tok.isspace() or tok == "":
            fixed_tokens.append(tok)
        else:
            fixed_tokens.append(_undouble_word_if_pattern(tok))
    return "".join(fixed_tokens)


# =========================
# HELPER: DICTIONARY-BASED GLUED WORD SPLITTING
# =========================

def is_real_word(word: str) -> bool:
    """
    Use wordfreq to decide if a string is a real English word.
    zipf_frequency > 1.5 means it's reasonably common.
    """
    return zipf_frequency(word, "en") > 1.5


def smart_split_glued_token(token: str):
    """
    Automatically split long glued tokens into valid words
    using English word frequency data.

    Example:
    'thesamecountryorevenon' → ['the', 'same', 'country', 'or', 'even', 'on']

    This is a greedy algorithm: tries the longest valid prefix first.
    """
    tok = token.lower()
    if len(tok) < 20:  # only bother for long glued tokens
        return [token]

    results = []

    def split_recursive(s: str):
        if not s:
            return

        # Try longest possible prefix first
        for i in range(len(s), 0, -1):
            piece = s[:i]
            if is_real_word(piece):
                results.append(piece)
                split_recursive(s[i:])
                return

        # if no prefix found, keep whole
        results.append(s)

    split_recursive(tok)
    return results


def split_glued_words(text: str) -> str:
    """
    Apply smart_split_glued_token to long lowercase tokens,
    preserving whitespace.
    """
    tokens = re.split(r"(\s+)", text)
    fixed = []
    for tok in tokens:
        if tok.isspace() or tok == "":
            fixed.append(tok)
        elif re.fullmatch(r"[a-z]+", tok) and len(tok) > 20:
            pieces = smart_split_glued_token(tok)
            fixed.append(" ".join(pieces))
        else:
            fixed.append(tok)
    return "".join(fixed)


# =========================
# TEXT CLEANING
# =========================

def basic_clean(text: str) -> str:
    """
    Clean raw PDF text to make it more readable and TF-IDF friendly:
    ✔ normalize spaces/newlines
    ✔ fix hyphen-line breaks
    ✔ join broken lines
    ✔ remove multi-number reference patterns like 1,3,6,7
    ✔ fix punctuation spacing
    ✔ split long glued tokens using dictionary-based splitting
    ✔ fix doubled-letter artifacts from some PDFs
    ✔ collapse extra spaces
    """
    if not text:
        return ""

    # Normalize special spaces & newlines
    text = text.replace("\r", "\n")
    text = text.replace("\xa0", " ")

    # 1) Fix hyphen at end of line:
    # "comput-\ning" → "computing"
    text = re.sub(r"-\s*\n\s*", "", text)

    # 2) Join lines that are actually in the same paragraph:
    # newline followed by lowercase/digit → treat as space
    # "problematic,\n but" → "problematic, but"
    text = re.sub(r"\n(?=[a-z0-9])", " ", text)

    # 3) Collapse multiple blank lines into paragraph breaks
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # 4) Remove reference-style number groups like "1,3,6,7"
    # after a period before a capital letter:
    # "worldwide.1,3,6,7 The" → "worldwide. The"
    text = re.sub(r"\.(\s*\d+(?:,\s*\d+)+)(?=\s+[A-Z])", ". ", text)

    # Also remove standalone grouped numbers between spaces:
    # "exposure 1,3,6,7 is" → "exposure is"
    text = re.sub(r"\s+\d+(?:,\s*\d+){1,}\s+", " ", text)

    # 5) Ensure there is a space after punctuation like , ; : if missing
    # "exposure is high,although" -> "exposure is high, although"
    text = re.sub(r"([,;:])(?=\S)", r"\1 ", text)

    # 6) Sometimes PDFs glue letters with capital in middle:
    # "UNEPGuidelines" → "UNEP Guidelines"
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    # 7) Split long glued lowercase tokens using dictionary-based splitting
    text = split_glued_words(text)

    # 8) Collapse extra spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 9) Finally fix doubled-letter artifacts like:
    # "SSyyrraaccuussee UUnniivveerrssiittyy" → "Syracuse University"
    text = _fix_doubled_words_in_text(text)

    return text.strip()


# =========================
# TF-IDF KEYWORD EXTRACTION
# =========================

def extract_keywords(text: str, top_n: int = 10):
    """
    Extract top_n keywords / keyphrases using TF-IDF from a single text.
    Uses unigrams + bigrams.
    Returns list of (term, score).
    """
    text = basic_clean(text)

    if not text or len(text.strip()) == 0:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",          # <-- use sklearn stopwords
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # only alphabetic tokens
    )

    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    top_n = min(top_n, len(scores))
    if top_n == 0:
        return []

    top_indices = scores.argsort()[-top_n:][::-1]
    keywords = [(feature_names[i], float(scores[i])) for i in top_indices]
    return keywords


# =========================
# TF-IDF SENTENCE-BASED SUMMARIZATION
# =========================

def summarize_text(text: str, num_sentences: int = 3):
    """
    Simple extractive summary:
    - Clean text
    - Split into sentences
    - Use TF-IDF on sentences
    - Score each sentence as sum of TF-IDF weights
    - Return top num_sentences sentences
    """
    text = basic_clean(text)
    if not text or len(text.strip()) == 0:
        return ""

    sentences = [s.strip() for s in sent_tokenize(text)]
    # Drop very short junk sentences
    sentences = [s for s in sentences if len(s) > 40]
    if not sentences:
        return ""

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(
        stop_words="english",          # <-- use sklearn stopwords
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)

    scores = tfidf_matrix.sum(axis=1).A1  # convert to 1D array

    num_sentences = min(num_sentences, len(sentences))
    top_indices = scores.argsort()[-num_sentences:][::-1]

    top_indices_sorted = sorted(top_indices)
    summary_sentences = [sentences[i] for i in top_indices_sorted]
    return " ".join(summary_sentences)


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("Research Paper Assistant")
st.write(
    "Upload a research paper PDF to get section-wise summaries and keywords "
    "using TF-IDF and basic NLP."
)

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

col1, col2 = st.columns(2)

with col1:
    top_n_keywords = st.slider("Number of keywords per section", 5, 25, 10)
with col2:
    summary_len = st.slider("Summary sentences per section", 2, 7, 3)

st.markdown("---")

if uploaded_file is not None:
    with st.spinner("Reading and analysing the PDF..."):
        full_text = extract_text_from_pdf(uploaded_file)
        full_text = basic_clean(full_text)
        sections = split_into_sections(full_text)

    if not sections:
        st.error("Could not detect any sections. Showing entire paper as one block.")
        sections = {"Full Paper": full_text}

    # GLOBAL OVERVIEW
    global_summary = summarize_text(full_text, num_sentences=5)
    global_keywords = extract_keywords(full_text, top_n=20)

    tab_overview, tab_sections, tab_keywords, tab_raw = st.tabs(
        ["Overview", "Section Summaries", "Section Keywords", "Cleaned Text"]
    )

    # -------------------------
    # OVERVIEW TAB
    # -------------------------
    with tab_overview:
        st.subheader("Overall Summary")
        st.write(global_summary)

        st.subheader("Global Important Keywords / Phrases")
        if global_keywords:
            st.table(
                {
                    "Keyword / Phrase": [k for k, _ in global_keywords],
                    "TF-IDF Score": [round(s, 4) for _, s in global_keywords],
                }
            )
        else:
            st.info("No keywords found (text might be too short or empty).")

        st.subheader("Detected Sections")
        st.write(", ".join(sections.keys()))

    # -------------------------
    # SECTION SUMMARIES TAB
    # -------------------------
    with tab_sections:
        st.subheader("Section-wise Summaries")

        for name, content in sections.items():
            if not content or len(content.strip()) == 0:
                continue

            st.markdown(f"### 📌 {name}")
            summary = summarize_text(content, num_sentences=summary_len)
            st.write(summary)
            st.markdown("---")

    # -------------------------
    # SECTION KEYWORDS TAB
    # -------------------------
    with tab_keywords:
        st.subheader("Section-wise Keywords")

        for name, content in sections.items():
            if not content or len(content.strip()) == 0:
                continue

            st.markdown(f"### 🔍 {name}")
            kws = extract_keywords(content, top_n=top_n_keywords)
            if kws:
                st.table(
                    {
                        "Keyword / Phrase": [k for k, _ in kws],
                        "TF-IDF Score": [round(s, 4) for _, s in kws],
                    }
                )
            else:
                st.write("_No keywords extracted for this section._")
            st.markdown("---")

    # -------------------------
    # CLEANED TEXT TAB
    # -------------------------
    with tab_raw:
        st.subheader("Cleaned Text (for debugging / report)")
        st.write(full_text[:20000])  # avoid dumping gigantic text

else:
    st.info("Upload a PDF to start.")
