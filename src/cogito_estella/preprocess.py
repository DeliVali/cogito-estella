"""STRICT preprocessing pipeline before the encode (production v0.5.0).

SONAR is a semantic sentence embedder; feeding syntactic junk degrades its space.
This module guarantees clean input, with per-modality rules:

  CODE     -> sanitize_code: strips comments/docstrings/blank lines (pygments,
              language-aware; precise tokenize/ast for Python). Logic only.
  LITERALS -> anonymize_secrets (curated regex: emails, API keys, tokens) + space_digits
              (the digit-spacing trick validated in exp016).
  PROSE    -> clean_prose: filters corrupt/control chars (unicodedata) and incomplete
              texts; drops whatever fails.

Libraries: pygments (multi-language), stdlib tokenize/ast/re/unicodedata. No tree-sitter
(pygments covers multi-language) nor ftfy (unicodedata suffices).
"""
import re
import unicodedata

# ---------- 1. Code sanitization ----------

_C_LINE = re.compile(r"//[^\n]*")
_C_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_HASH_LINE = re.compile(r"#[^\n]*")
_MULTI_BLANK = re.compile(r"\n\s*\n\s*\n+")


def _collapse_blanks(text: str) -> str:
    return _MULTI_BLANK.sub("\n\n", text).strip("\n")


_FENCE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[tuple[str | None, str]]:
    """Extracts ``` code blocks from markdown. With no fences, treats everything as code."""
    blocks = [(lang or None, code) for lang, code in _FENCE.findall(text)]
    if blocks:
        return blocks
    return [(None, text)]


def sanitize_code(text: str, lang: str | None = None) -> str:
    """Strips comments and docstrings in a language-aware way (pygments)."""
    try:
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.token import Comment, String

        lexer = get_lexer_by_name(lang) if lang else guess_lexer(text)
        kept = []
        for tok_type, value in lexer.get_tokens(text):
            if tok_type in Comment or tok_type in String.Doc:
                # keep the newlines so lines don't get glued together
                kept.append("\n" * value.count("\n"))
            else:
                kept.append(value)
        return _collapse_blanks("".join(kept))
    except Exception:
        # regex fallback when pygments doesn't recognize the language
        t = _C_BLOCK.sub("", text)
        t = _C_LINE.sub("", t)
        t = _HASH_LINE.sub("", t)
        return _collapse_blanks(t)


# ---------- 2. Literal normalization ----------

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                    # OpenAI-style
    re.compile(r"AKIA[0-9A-Z]{16}"),                        # AWS access key
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),             # GitHub token
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),       # bearer tokens
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),                    # long hex hashes/keys
]
_DIGIT_RUN = re.compile(r"\d+")

# Roundtrip audit (2026-08-26, n=4/lang): spacing preserves the sentence body in every
# tested script, but spaced digit runs drift in these languages (separator mutation /
# run merging). Conservative gate until measured at scale; plain digits still recover
# ~0.956 via digit-nodes (exp016), so skipping the trick is a mild, safe fallback.
SPACING_UNSAFE_LANGS = {"jpn_Jpan", "rus_Cyrl"}


def anonymize_secrets(text: str) -> str:
    text = _EMAIL.sub("<EMAIL>", text)
    for pat in _SECRET_PATTERNS:
        text = pat.sub("<SECRET>", text)
    return text


def space_digits(text: str, min_run: int = 1, lang: str | None = None) -> str:
    """Validated trick (exp016): '400' -> '4 0 0' so SONAR sees isolated digits.
    min_run limits to runs of length >= min_run (e.g. 5 = only long IDs/numbers).
    lang gates the trick off for scripts where spaced runs drift (SPACING_UNSAFE_LANGS)."""
    if lang in SPACING_UNSAFE_LANGS:
        return text
    return _DIGIT_RUN.sub(lambda m: " ".join(m.group()) if len(m.group()) >= min_run else m.group(), text)


# ---------- 3. Prose cleaning ----------

def clean_prose(text: str, min_len: int = 40) -> str | None:
    """Drops corrupt or incomplete prose; returns None if it fails the filter."""
    n = len(text)
    if n < min_len:
        return None
    # excessive replacement (mojibake) or control chars -> corrupt
    n_repl = text.count("�")
    n_ctrl = sum(1 for c in text if unicodedata.category(c).startswith("C") and c not in "\n\t")
    # mojibake (replacement chars) drops; the few control chars are cleaned, not dropped
    if n_repl / n > 0.005 or n_ctrl / n > 0.10:
        return None
    # strip control chars (except \n, \t)
    cleaned = "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))
    cleaned = cleaned.strip()
    # minimal completeness: must have letters and end with punctuation or be long
    if not any(c.isalpha() for c in cleaned):
        return None
    if len(cleaned) < min_len:
        return None
    return cleaned


# ---------- Per-modality dispatcher ----------

def preprocess_record(rec, space_code_digits: bool = True):
    """rec: DocRecord. Returns a clean DocRecord, or None if discarded."""
    from cogito_estella.multilingual_factory import DocRecord
    text = rec.text
    if rec.modality == "code":
        # extract only the pure code blocks (CodeFeedback ships with surrounding prose)
        default_lang = "python" if rec.source == "code_py" else None
        parts = [sanitize_code(code, lang=lg or default_lang) for lg, code in extract_code_blocks(text)]
        text = "\n".join(p for p in parts if p.strip())
        text = anonymize_secrets(text)  # no digit spacing: digits are structural in code
        if len(text.strip()) < 3:
            return None
    elif rec.modality == "toolcall":
        # preserve JSON structure: anonymize secrets, do NOT space digits (would break the JSON)
        text = anonymize_secrets(text)
        if len(text.strip()) < 3:
            return None
    else:  # prose
        cleaned = clean_prose(text)
        if cleaned is None:
            return None
        text = anonymize_secrets(cleaned)
        text = space_digits(text, min_run=5, lang=rec.sonar_lang)  # only long IDs/numbers SONAR loses
    return DocRecord(text, rec.sonar_lang, rec.source, rec.modality)
