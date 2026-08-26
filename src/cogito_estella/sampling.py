"""Muestreo de unidades de texto por categoría: sintéticos, HF streaming y fallback local."""
import random
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    category: str
    text: str
    source: str
    lang: str


_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÑ¿¡\"'(])")

_NUM_TEMPLATES = [
    ("El paquete pesa {a}.{b} kg y llegará el {d} de marzo de 20{y}.", "spa_Latn"),
    ("La factura #{i} suma ${a},{b}{b}0.50 con IVA del {t}%.", "spa_Latn"),
    ("El tratamiento requiere {t} mg cada {h} horas durante {d} días.", "spa_Latn"),
    ("The sensor reported {a}.{b}°C at {h}:{m2} UTC on day {d}.", "eng_Latn"),
    ("Invoice {i} totals €{a}{b}.{t} due in {d} days.", "eng_Latn"),
    ("Model v{a}.{b}.{t} reduced latency from {i} ms to {d} ms.", "eng_Latn"),
]

_TOOL_NAMES = ["search_web", "get_weather", "send_email", "create_event", "query_db", "run_code"]


def synthetic_numbers(n: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        tpl, lang = rng.choice(_NUM_TEMPLATES)
        text = tpl.format(
            a=rng.randint(1, 99), b=rng.randint(0, 9), d=rng.randint(1, 28),
            y=rng.randint(24, 29), i=rng.randint(1000, 99999), t=rng.randint(2, 500),
            h=rng.randint(0, 23), m2=f"{rng.randint(0, 59):02d}",
        )
        out.append(Sample("numeros", text, "synthetic", lang))
    return out


def synthetic_json_tools(n: int, seed: int) -> list[Sample]:
    import json

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        name = rng.choice(_TOOL_NAMES)
        payload = {
            "name": name,
            "arguments": {
                "query": rng.choice(["clima CDMX", "flights to Madrid", "SELECT * FROM users", "reunión lunes"]),
                "limit": rng.randint(1, 50),
                "verbose": rng.choice([True, False]),
            },
        }
        out.append(Sample("json_tools", json.dumps(payload, ensure_ascii=False), "synthetic", "eng_Latn"))
    return out


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_RE.split(text) if p.strip()]


def good_unit(text: str, min_chars: int = 30, max_chars: int = 256) -> bool:
    return min_chars <= len(text) <= max_chars


def sample_hf_prose(dataset: str, config: str | None, text_field: str, n: int, seed: int,
                     category: str, lang: str) -> list[Sample]:
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split="train", streaming=True)
    rng = random.Random(seed)
    out: list[Sample] = []
    for doc in ds:
        for sent in split_sentences(doc[text_field]):
            if good_unit(sent) and rng.random() < 0.3:
                out.append(Sample(category, sent, dataset, lang))
                if len(out) >= n:
                    return out
    return out


def sample_local_code(root: str, n: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    fragments: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        text = path.read_text(errors="ignore")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for i in range(0, max(len(lines) - 2, 0)):
            frag = "\n".join(lines[i : i + 3])
            if good_unit(frag, min_chars=20):
                fragments.append(frag)
    rng.shuffle(fragments)
    return [Sample("codigo", f, "local_repo", "eng_Latn") for f in fragments[:n]]
