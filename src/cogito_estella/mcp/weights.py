"""Encoder-aware weight resolution: explicit paths, or the published assets of one
encoder from Hugging Face (manifest `encoders.json` plus a per-encoder subfolder)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from cogito_estella.encoders import DEFAULT_ENCODER, ensemble_operating_point

HF_REPO = "DeliVali/cogito-estella"
MANIFEST = "encoders.json"
DEFAULT_CHECKPOINTS = ("cogito-prose-ontology.pt", "cogito-prose-ontology-s2.pt",
                       "cogito-prose-ontology-s3.pt")
DEFAULT_VOCAB = "vocab-onto.json"

# Published assets per encoder: the files to fetch and the decode thresholds they were
# validated under. `subfolder` is None for the SONAR-era files at the repository root;
# the manifest may override any of these entries.
ENCODER_ASSETS: dict[str, dict] = {
    "sonar": {
        "checkpoints": DEFAULT_CHECKPOINTS,
        "vocab": DEFAULT_VOCAB,
        "pool": None,
        "subfolder": None,
        "operating_point": ensemble_operating_point("sonar"),
    },
    "m2m100-pool": {
        "checkpoints": ("cogito-prose-ontology-m2mpool.safetensors",
                        "cogito-prose-ontology-m2mpool-s2.safetensors",
                        "cogito-prose-ontology-m2mpool-s3.safetensors"),
        "vocab": "vocab-onto-m2mpool.json",
        "pool": "pool.safetensors",
        "subfolder": "m2m100-pool",
        "operating_point": ensemble_operating_point("m2m100-pool"),
    },
}

_HELP = ("Get them with `--checkpoint <file> --vocab <file>` (local paths) or let cogito-mcp "
         f"download the defaults from https://huggingface.co/{HF_REPO} (needs network; "
         "omit --no-download).")


class WeightsError(Exception):
    pass


@dataclass(frozen=True)
class ResolvedWeights:
    """Files the server hands to the extractor, plus what the publisher says about them.

    `encoder` is None when explicit paths were resolved without `--encoder`: the
    checkpoints decide the semantic space, so naming one here would announce a space
    the extractor may not use. `operating_point` is the published pair of decode
    thresholds, None when the publisher names none.
    """

    checkpoints: list[Path]
    vocab: Path
    pool: Path | None
    encoder: str | None
    operating_point: tuple[float, float] | None = None


def _hf_download(repo_id: str, filename: str, local_files_only: bool = False,
                 subfolder: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, subfolder=subfolder,
                           local_files_only=local_files_only)


def load_manifest(download: bool = True, downloader=None) -> dict:
    """The published `encoders.json`, or {} when it is not there. A manifest that IS
    there and is unreadable is a defect, not an absence."""
    dl = downloader or _hf_download
    try:
        path = dl(HF_REPO, MANIFEST, local_files_only=not download, subfolder=None)
    except OSError:                                # cache miss offline, network error, 404
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WeightsError(f"manifest {MANIFEST} is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise WeightsError(f"manifest {MANIFEST} is not a mapping of encoder -> assets")
    return data


def _assets(encoder: str, manifest: dict) -> dict:
    """Built-in table for `encoder`, overridden by its manifest entry when present.

    Only `checkpoints`, `vocab`, `pooling`, `subfolder` and `operating_point` govern
    what happens here; a manifest's `revision`, `dim`, `normalize` and `licence` are
    descriptive — the binding values travel in the checkpoint sidecars, which the
    extractor checks against the active encoder.
    """
    entry = manifest.get(encoder)
    if entry is None and encoder not in ENCODER_ASSETS:
        raise WeightsError(f"no published weights for encoder {encoder!r}; published: "
                           f"{', '.join(sorted(ENCODER_ASSETS))}. {_HELP}")
    assets = dict(ENCODER_ASSETS.get(encoder, {"checkpoints": (), "vocab": None,
                                               "pool": None, "subfolder": None}))
    if isinstance(entry, dict):
        for key, name in (("checkpoints", "checkpoints"), ("vocab", "vocab"),
                          ("pool", "pooling"), ("subfolder", "subfolder"),
                          ("operating_point", "operating_point")):
            if name in entry:
                assets[key] = entry[name]
    if not assets.get("checkpoints") or not assets.get("vocab"):
        raise WeightsError(f"manifest entry for encoder {encoder!r} names no "
                           f"checkpoints and vocabulary. {_HELP}")
    return assets


def _operating_point(encoder: str, assets: dict) -> tuple[float, float] | None:
    """The published pair, validated: a manifest is remote data, not a literal."""
    point = assets.get("operating_point")
    if point is None:
        return None
    try:
        existence, adjacency = (float(x) for x in point)
    except (TypeError, ValueError) as exc:
        raise WeightsError(f"operating_point for encoder {encoder!r} is not a pair of "
                           f"numbers: {point!r}") from exc
    return existence, adjacency


def resolve(checkpoints: list[str] | None, vocab: str | None, download: bool = True,
            downloader=None, encoder: str | None = None,
            pool: str | Path | None = None) -> ResolvedWeights:
    """Explicit paths win and must all exist; otherwise the published assets of
    `encoder` (default `DEFAULT_ENCODER`). Pooling weights are independent: an explicit
    `--pool` is always used."""
    if bool(checkpoints) != bool(vocab):
        raise WeightsError("pass both --checkpoint and --vocab, or neither")
    # Only the download branch may fall back to the default: with explicit paths the
    # checkpoints decide the space, so there is no encoder to name.
    named = encoder or (None if checkpoints else DEFAULT_ENCODER)
    where = f" for encoder {named}" if named else ""
    pool_path = Path(pool) if pool else None
    if pool_path and not pool_path.is_file():
        raise WeightsError(f"missing pooling weights{where}: {pool_path}")
    if checkpoints:
        paths = [Path(c) for c in checkpoints] + [Path(vocab)]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise WeightsError(f"missing weight files{where}: {', '.join(missing)}")
        return ResolvedWeights(paths[:-1], paths[-1], pool_path, encoder)

    encoder = named
    dl = downloader or _hf_download
    assets = _assets(encoder, load_manifest(download, dl))
    wanted = [*assets["checkpoints"], assets["vocab"]]
    if assets.get("pool") and pool_path is None:
        wanted.append(assets["pool"])
    got, missing = [], []
    for name in wanted:
        try:
            got.append(Path(dl(HF_REPO, name, local_files_only=not download,
                               subfolder=assets.get("subfolder"))))
        except OSError:                            # cache miss offline, network error, 404
            missing.append(name)
    if missing:
        raise WeightsError(f"weights for encoder {encoder} not available: "
                           f"{', '.join(missing)}. {_HELP}")
    n_ck = len(assets["checkpoints"])
    if pool_path is None and assets.get("pool"):
        pool_path = got[-1]
    return ResolvedWeights(got[:n_ck], got[n_ck], pool_path, encoder,
                           _operating_point(encoder, assets))


def ensure_spacy_model(download: bool = True, model: str = "en_core_web_sm") -> None:
    """The parser model is not on PyPI: download once, or explain the command."""
    import spacy.util
    if spacy.util.is_package(model):
        return
    if not download:
        raise WeightsError(f"spaCy model {model} missing; run: python -m spacy download {model}")
    print(f"cogito-mcp: downloading spaCy model {model}", file=sys.stderr)
    from spacy.cli import download as spacy_download
    spacy_download(model)
