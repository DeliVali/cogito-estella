"""Checkpoint and vocabulary resolution: explicit paths, or the published ontology
ensemble from Hugging Face (cached by huggingface_hub)."""
from __future__ import annotations

import sys
from pathlib import Path

HF_REPO = "DeliVali/cogito-estella"
DEFAULT_CHECKPOINTS = ("cogito-prose-ontology.pt", "cogito-prose-ontology-s2.pt",
                       "cogito-prose-ontology-s3.pt")
DEFAULT_VOCAB = "vocab-onto.json"
_HELP = ("Get them with `--checkpoint <file> --vocab <file>` (local paths) or let cogito-mcp "
         f"download the defaults from https://huggingface.co/{HF_REPO} (needs network; "
         "omit --no-download).")


class WeightsError(Exception):
    pass


def _hf_download(repo_id: str, filename: str, local_files_only: bool = False) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, local_files_only=local_files_only)


def resolve(checkpoints: list[str] | None, vocab: str | None, download: bool = True,
            downloader=None) -> tuple[list[Path], Path]:
    """Explicit paths win and must all exist; otherwise the Hugging Face defaults."""
    if bool(checkpoints) != bool(vocab):
        raise WeightsError("pass both --checkpoint and --vocab, or neither")
    if checkpoints:
        paths = [Path(c) for c in checkpoints] + [Path(vocab)]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise WeightsError(f"missing weight files: {', '.join(missing)}")
        return paths[:-1], paths[-1]
    dl = downloader or _hf_download
    got, missing = [], []
    for name in (*DEFAULT_CHECKPOINTS, DEFAULT_VOCAB):
        try:
            got.append(Path(dl(HF_REPO, name, local_files_only=not download)))
        except Exception:  # noqa: BLE001 - downloader raises various types (network, 404, cache)
            missing.append(name)
    if missing:
        raise WeightsError(f"default weights not available: {', '.join(missing)}. {_HELP}")
    return got[:-1], got[-1]


def ensure_spacy_model(download: bool = True, model: str = "en_core_web_sm") -> None:
    """The parser model is not on PyPI: download once, or explain the command."""
    import spacy.util
    if spacy.util.is_package(model):
        return
    if not download:
        raise WeightsError(f"spaCy model {model} missing; run: python -m spacy download {model}")
    print(f"cogito-mcp: downloading spaCy model {model}", file=sys.stderr)
    from spacy.cli import download
    download(model)
