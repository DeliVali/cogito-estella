"""Fábrica de conceptos MULTILINGÜE con FORMATO CANÓNICO NORMALIZADO (producción v0.5.0).

Toda fuente heterogénea (prosa multilingüe, código, tool-calls) se normaliza a un mismo
`DocRecord` antes de segmentar/encodear. Agregar una fuente nueva = una entrada en MIX +,
si hace falta, un adaptador de campo. Extensible por diseño: "agregar más y más información".

Separación clave: el código para CARGAR el dataset de HF (`hf_config`, p. ej. cmn_Hani) es
distinto del tag con que SONAR ENCODEA (`sonar_lang`, p. ej. zho_Hans). El formato lo separa.

La `modality` gobierna la segmentación:
  prose/code -> SaT (oraciones); toolcall -> la unidad estructurada completa (1 concepto).

Mezcla congelada 75/15/10 (% de docs).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str        # dataset HF, o "__synthetic__"
    hf_config: str | None   # config para CARGAR (cmn_Hani); None si no aplica
    field: str | None       # campo de texto en el registro HF
    sonar_lang: str         # tag con que SONAR ENCODEA (zho_Hans)
    modality: str           # "prose" | "code" | "toolcall"
    pct: int


@dataclass
class DocRecord:
    """Formato canónico: toda fuente se normaliza a esto antes de la fábrica."""
    text: str
    sonar_lang: str
    source: str
    modality: str


MIX: list[SourceSpec] = [
    # prosa multilingüe (75%)
    SourceSpec("en", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text", "eng_Latn", "prose", 22),
    SourceSpec("es", "HuggingFaceFW/fineweb-2", "spa_Latn", "text", "spa_Latn", "prose", 14),
    SourceSpec("fr", "HuggingFaceFW/fineweb-2", "fra_Latn", "text", "fra_Latn", "prose", 8),
    SourceSpec("de", "HuggingFaceFW/fineweb-2", "deu_Latn", "text", "deu_Latn", "prose", 7),
    SourceSpec("pt", "HuggingFaceFW/fineweb-2", "por_Latn", "text", "por_Latn", "prose", 6),
    SourceSpec("it", "HuggingFaceFW/fineweb-2", "ita_Latn", "text", "ita_Latn", "prose", 5),
    SourceSpec("nl", "HuggingFaceFW/fineweb-2", "nld_Latn", "text", "nld_Latn", "prose", 4),
    SourceSpec("ru", "HuggingFaceFW/fineweb-2", "rus_Cyrl", "text", "rus_Cyrl", "prose", 4),
    SourceSpec("zh", "HuggingFaceFW/fineweb-2", "cmn_Hani", "text", "zho_Hans", "prose", 3),  # carga cmn_Hani, encodea zho_Hans
    SourceSpec("ar", "HuggingFaceFW/fineweb-2", "arb_Arab", "text", "arb_Arab", "prose", 2),
    # código (15%) — SONAR trata el código como texto (tag eng_Latn)
    SourceSpec("code_py", "codeparrot/codeparrot-clean-valid", None, "content", "eng_Latn", "code", 8),
    SourceSpec("code_multi", "m-a-p/CodeFeedback-Filtered-Instruction", None, "answer", "eng_Latn", "code", 7),
    # agéntico (10%)
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
    """Segmentación según modalidad. toolcall corto -> 1 unidad estructurada;
    prose/code y toolcall largo -> SaT. Acota unidades por doc para memoria."""
    if rec.modality == "toolcall" and len(rec.text) <= 256:
        return [rec.text.strip()]
    units = segmenter.segment(rec.text)
    return units[:max_units]


def iter_records(src: SourceSpec, n: int, seed: int):
    """Normaliza una fuente a un flujo de DocRecords canónicos."""
    if src.dataset == "__synthetic__":
        from cogito_estella import sampling
        for smp in sampling.synthetic_json_tools(n, seed):
            yield DocRecord(smp.text, src.sonar_lang, src.name, src.modality)
        return
    from datasets import load_dataset
    ds = load_dataset(src.dataset, src.hf_config, split="train", streaming=True) if src.hf_config \
        else load_dataset(src.dataset, split="train", streaming=True)
    count = 0
    # umbral de longitud según modalidad: los tool-calls pueden ser cortos
    min_len = 30 if src.modality == "toolcall" else 120
    for doc in ds:
        text = _extract_field(doc, src.field)
        if text and len(text) >= min_len:
            yield DocRecord(text, src.sonar_lang, src.name, src.modality)
            count += 1
            if count >= n:
                return
