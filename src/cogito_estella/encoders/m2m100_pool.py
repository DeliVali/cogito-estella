"""m2m100-pool adapter: frozen M2M-100 encoder (MIT) + the learned attention pooling.

Native output is the raw pooled vector: the trunk is trained on it, so `normalize=None`
must not rescale. `normalize=True` gives unit vectors for retrieval.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from cogito_estella.encoders.pooling import DIM, PoolWeightsError, load_pool, sha256_file

M2M100_MODEL = "facebook/m2m100_418M"
M2M100_REVISION = "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
POOL_ENV = "COGITO_POOL"
_MAX_LEN = 128                                     # training truncated at 64; 0.05 % exceed it

# SONAR-style codes -> M2M-100 tokenizer `src_lang`.
LANG_TO_M2M = {
    "eng_Latn": "en", "spa_Latn": "es", "fra_Latn": "fr", "deu_Latn": "de",
    "por_Latn": "pt", "ita_Latn": "it", "zho_Hans": "zh", "rus_Cyrl": "ru",
    "arb_Arab": "ar", "jpn_Jpan": "ja",
}
_WARNED: set[str] = set()


def lang_to_m2m(lang: str) -> str:
    """Unknown codes fall back to English, warned once each rather than every batch."""
    code = LANG_TO_M2M.get(lang)
    if code is not None:
        return code
    if lang not in _WARNED:
        _WARNED.add(lang)
        print(f"m2m100-pool: unknown language {lang!r}; encoding as 'en'", file=sys.stderr)
    return "en"


class M2m100PoolEncoder:
    """Frozen M2M-100 token states pooled by a trained AttnPool, behind TextEncoder."""

    name = "m2m100-pool"
    dim = DIM
    native_normalized = False

    def __init__(self, device: str | None = None, download: bool = True,
                 pool_path: str | Path | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.download = download
        self.pool_path = self._resolve_pool_path(pool_path)
        self.pool_sha256 = sha256_file(self.pool_path)
        self.revision = f"{M2M100_REVISION}+pool:{self.pool_sha256[:12]}"
        self._model = None
        self._tok = None
        self._pool = None

    @staticmethod
    def _resolve_pool_path(pool_path: str | Path | None) -> Path:
        """Explicit argument > env; the weights identify the encoder, so never guess."""
        candidate = pool_path or os.environ.get(POOL_ENV)
        if not candidate:
            raise PoolWeightsError(
                "m2m100-pool needs learned pooling weights: pass pool_path (--pool "
                f"<pool.pt>) or set {POOL_ENV}=<pool.pt>")
        path = Path(candidate)
        if not path.is_file():
            raise PoolWeightsError(f"pooling weights not found: {path}")
        return path

    def _load_pool(self):
        if self._pool is None:
            pool = load_pool(self.pool_path, self.device)
            if pool.d != DIM:
                raise PoolWeightsError(
                    f"{self.pool_path} pools to {pool.d} dims, the encoder contract is {DIM}")
            self._pool = pool
        return self._pool

    def _load(self):
        if self._model is None:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

            kwargs = {"revision": M2M100_REVISION, "local_files_only": not self.download}
            self._tok = M2M100Tokenizer.from_pretrained(M2M100_MODEL, src_lang="en", **kwargs)
            model = M2M100ForConditionalGeneration.from_pretrained(
                M2M100_MODEL, **kwargs).get_encoder().to(self.device)
            if str(self.device).startswith("cuda"):
                model = model.half()
            self._model = model.eval().requires_grad_(False)
        return self._tok, self._model, self._load_pool()

    @torch.inference_mode()
    def encode(self, texts, lang: str = "eng_Latn", batch_size: int = 64,
               normalize: bool | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        tok, model, pool = self._load()
        tok.src_lang = lang_to_m2m(lang)
        rows = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=_MAX_LEN, return_tensors="pt").to(self.device)
            mask = enc["attention_mask"]
            hidden = model(input_ids=enc["input_ids"], attention_mask=mask).last_hidden_state
            vec = pool(hidden.float(), mask.bool()).float()
            if normalize:
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            rows.append(vec.cpu().numpy())
        return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
