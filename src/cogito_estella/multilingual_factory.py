"""MULTILINGUAL concept factory with a NORMALIZED CANONICAL FORMAT (production v0.5.0).

Every heterogeneous source (multilingual prose, code, tool-calls) is normalized to a
single `DocRecord` before segmenting/encoding. Adding a new source = one MIX entry +,
if needed, a field adapter. Extensible by design: "add more and more information".

Key separation: the code that LOADS the HF dataset (`hf_config`, e.g. cmn_Hani) differs
from the tag SONAR ENCODES with (`sonar_lang`, e.g. zho_Hans). The format keeps them apart.

The `modality` governs segmentation:
  prose/code -> SaT (sentences); toolcall -> the full structured unit (1 concept).

Frozen 75/15/10 mix (% of docs).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str        # HF dataset, or "__synthetic__"
    hf_config: str | None   # config to LOAD (cmn_Hani); None if N/A
    field: str | None       # text field in the HF record
    sonar_lang: str         # tag SONAR ENCODES with (zho_Hans)
    modality: str           # "prose" | "code" | "toolcall"
    pct: int


@dataclass
class DocRecord:
    """Canonical format: every source is normalized to this before the factory."""
    text: str
    sonar_lang: str
    source: str
    modality: str


MIX: list[SourceSpec] = [
    # multilingual prose (75%)
    SourceSpec("en", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text", "eng_Latn", "prose", 22),
    SourceSpec("es", "HuggingFaceFW/fineweb-2", "spa_Latn", "text", "spa_Latn", "prose", 14),
    SourceSpec("fr", "HuggingFaceFW/fineweb-2", "fra_Latn", "text", "fra_Latn", "prose", 8),
    SourceSpec("de", "HuggingFaceFW/fineweb-2", "deu_Latn", "text", "deu_Latn", "prose", 7),
    SourceSpec("pt", "HuggingFaceFW/fineweb-2", "por_Latn", "text", "por_Latn", "prose", 6),
    SourceSpec("it", "HuggingFaceFW/fineweb-2", "ita_Latn", "text", "ita_Latn", "prose", 5),
    SourceSpec("nl", "HuggingFaceFW/fineweb-2", "nld_Latn", "text", "nld_Latn", "prose", 4),
    SourceSpec("ru", "HuggingFaceFW/fineweb-2", "rus_Cyrl", "text", "rus_Cyrl", "prose", 4),
    SourceSpec("zh", "HuggingFaceFW/fineweb-2", "cmn_Hani", "text", "zho_Hans", "prose", 3),  # loads cmn_Hani, encodes zho_Hans
    SourceSpec("ar", "HuggingFaceFW/fineweb-2", "arb_Arab", "text", "arb_Arab", "prose", 2),
    # code (15%) — SONAR treats code as text (eng_Latn tag)
    SourceSpec("code_py", "codeparrot/codeparrot-clean-valid", None, "content", "eng_Latn", "code", 8),
    SourceSpec("code_multi", "m-a-p/CodeFeedback-Filtered-Instruction", None, "answer", "eng_Latn", "code", 7),
    # agentic (10%)
    SourceSpec("toolcalls", "glaiveai/glaive-function-calling-v2", None, "chat", "eng_Latn", "toolcall", 6),
    SourceSpec("toolcalls_syn", "__synthetic__", None, None, "eng_Latn", "toolcall", 4),
]


def sources() -> list[SourceSpec]:
    return list(MIX)


def doc_targets(total_docs: int) -> dict:
    return {s.name: round(total_docs * s.pct / 100) for s in MIX}


def _extract_field(doc: dict, field: str) -> str:
    val = doc.get(field)
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return " ".join(str(x) for x in val)
    return str(val) if val is not None else ""


def segment_record(rec: DocRecord, segmenter, max_units: int = 64) -> list[str]:
    """Segmentation by modality. short toolcall -> 1 structured unit;
    prose/code and long toolcall -> SaT. Caps units per doc for memory."""
    if rec.modality == "toolcall" and len(rec.text) <= 256:
        return [rec.text.strip()]
    units = segmenter.segment(rec.text)
    return units[:max_units]


def iter_records(src: SourceSpec, n: int, seed: int):
    """Normalizes a source to a stream of canonical DocRecords."""
    if src.dataset == "__synthetic__":
        from cogito_estella import sampling
        for smp in sampling.synthetic_json_tools(n, seed):
            yield DocRecord(smp.text, src.sonar_lang, src.name, src.modality)
        return
    from datasets import load_dataset
    ds = load_dataset(src.dataset, src.hf_config, split="train", streaming=True) if src.hf_config \
        else load_dataset(src.dataset, split="train", streaming=True)
    count = 0
    # length threshold by modality: tool-calls can be short
    min_len = 30 if src.modality == "toolcall" else 120
    for doc in ds:
        text = _extract_field(doc, src.field)
        if text and len(text) >= min_len:
            yield DocRecord(text, src.sonar_lang, src.name, src.modality)
            count += 1
            if count >= n:
                return
