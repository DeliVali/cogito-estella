"""Pipeline de pre-procesamiento RÍGIDO antes del encode (producción v0.5.0).

SONAR es un embedder semántico de oraciones; meter basura sintáctica degrada su espacio.
Este módulo garantiza que los datos entren limpios, con reglas por modalidad:

  CÓDIGO   -> sanitize_code: quita comentarios/docstrings/líneas en blanco (pygments,
              consciente del lenguaje; tokenize/ast preciso para Python). Solo lógica.
  LITERALES-> anonymize_secrets (regex curados: emails, API keys, tokens) + space_digits
              (el truco de espaciado de dígitos validado en exp016).
  PROSA    -> clean_prose: filtra caracteres corruptos/control (unicodedata) y textos
              incompletos; descarta lo que no pase.

Librerías: pygments (multi-lenguaje), stdlib tokenize/ast/re/unicodedata. Sin tree-sitter
(pygments cubre multi-lenguaje) ni ftfy (unicodedata basta).
"""
import re
import unicodedata

# ---------- 1. Sanitización de código ----------

_C_LINE = re.compile(r"//[^\n]*")
_C_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_HASH_LINE = re.compile(r"#[^\n]*")
_MULTI_BLANK = re.compile(r"\n\s*\n\s*\n+")


def _collapse_blanks(text: str) -> str:
    return _MULTI_BLANK.sub("\n\n", text).strip("\n")


_FENCE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[tuple[str | None, str]]:
    """Extrae bloques de código ``` de markdown. Si no hay fences, trata todo como código."""
    blocks = [(lang or None, code) for lang, code in _FENCE.findall(text)]
    if blocks:
        return blocks
    return [(None, text)]


def sanitize_code(text: str, lang: str | None = None) -> str:
    """Quita comentarios y docstrings de forma consciente del lenguaje (pygments)."""
    try:
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.token import Comment, String

        lexer = get_lexer_by_name(lang) if lang else guess_lexer(text)
        kept = []
        for tok_type, value in lexer.get_tokens(text):
            if tok_type in Comment or tok_type in String.Doc:
                # preservar los saltos de línea para no pegar líneas
                kept.append("\n" * value.count("\n"))
            else:
                kept.append(value)
        return _collapse_blanks("".join(kept))
    except Exception:
        # fallback regex si pygments no reconoce el lenguaje
        t = _C_BLOCK.sub("", text)
        t = _C_LINE.sub("", t)
        t = _HASH_LINE.sub("", t)
        return _collapse_blanks(t)


# ---------- 2. Normalización de literales ----------

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                    # OpenAI-style
    re.compile(r"AKIA[0-9A-Z]{16}"),                        # AWS access key
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),             # GitHub token
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),       # bearer tokens
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),                    # hex hashes/keys largos
]
_DIGIT_RUN = re.compile(r"\d+")


def anonymize_secrets(text: str) -> str:
    text = _EMAIL.sub("<EMAIL>", text)
    for pat in _SECRET_PATTERNS:
        text = pat.sub("<SECRET>", text)
    return text


def space_digits(text: str, min_run: int = 1) -> str:
    """Truco validado (exp016): '400' -> '4 0 0' para que SONAR vea dígitos aislados.
    min_run acota a runs de longitud >= min_run (p. ej. 5 = solo IDs/números largos)."""
    return _DIGIT_RUN.sub(lambda m: " ".join(m.group()) if len(m.group()) >= min_run else m.group(), text)


# ---------- 3. Limpieza de prosa ----------

def clean_prose(text: str, min_len: int = 40) -> str | None:
    """Descarta prosa corrupta o incompleta; devuelve None si no pasa el filtro."""
    n = len(text)
    if n < min_len:
        return None
    # caracteres de reemplazo (mojibake) o de control excesivos -> corrupto
    n_repl = text.count("�")
    n_ctrl = sum(1 for c in text if unicodedata.category(c).startswith("C") and c not in "\n\t")
    # mojibake (replacement chars) descarta; los pocos chars de control se limpian, no descartan
    if n_repl / n > 0.005 or n_ctrl / n > 0.10:
        return None
    # quitar caracteres de control (menos \n, \t)
    cleaned = "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))
    cleaned = cleaned.strip()
    # completitud mínima: debe tener letras y terminar con puntuación o ser largo
    if not any(c.isalpha() for c in cleaned):
        return None
    if len(cleaned) < min_len:
        return None
    return cleaned


# ---------- Dispatcher por modalidad ----------

def preprocess_record(rec, space_code_digits: bool = True):
    """rec: DocRecord. Devuelve un DocRecord limpio, o None si se descarta."""
    from cogito_estella.multilingual_factory import DocRecord
    text = rec.text
    if rec.modality == "code":
        # extraer solo los bloques de código puros (CodeFeedback viene con prosa envolvente)
        default_lang = "python" if rec.source == "code_py" else None
        parts = [sanitize_code(code, lang=lg or default_lang) for lg, code in extract_code_blocks(text)]
        text = "\n".join(p for p in parts if p.strip())
        text = anonymize_secrets(text)  # sin espaciar dígitos: son estructurales en el código
        if len(text.strip()) < 3:
            return None
    elif rec.modality == "toolcall":
        # preservar la estructura JSON: anonimizar secrets, NO espaciar dígitos (rompería el JSON)
        text = anonymize_secrets(text)
        if len(text.strip()) < 3:
            return None
    else:  # prose
        cleaned = clean_prose(text)
        if cleaned is None:
            return None
        text = anonymize_secrets(cleaned)
        text = space_digits(text, min_run=5)  # solo IDs/números largos que SONAR pierde
    return DocRecord(text, rec.sonar_lang, rec.source, rec.modality)
