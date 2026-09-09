"""BGE-M3 adapter (MIT weights) over transformers: CLS pooling, L2 norm, max length 512."""
from __future__ import annotations

import numpy as np
import torch

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_DIM = 1024
_MAX_LEN = 512


class BgeM3Encoder:
    """Dense BGE-M3 embeddings behind the TextEncoder contract.

    `lang` is ignored (the model is multilingual without a language token) and
    `normalize=False` is ignored: unit-norm CLS is this encoder's native contract.
    """

    name = "bge-m3"
    revision = BGE_M3_REVISION
    dim = _DIM
    native_normalized = True

    def __init__(self, device: str | None = None, download: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.download = download
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            kwargs = {"revision": BGE_M3_REVISION, "local_files_only": not self.download}
            self._tok = AutoTokenizer.from_pretrained(BGE_M3_MODEL, **kwargs)
            model = AutoModel.from_pretrained(BGE_M3_MODEL, **kwargs).to(self.device)
            if str(self.device).startswith("cuda"):
                model = model.half()
            self._model = model.eval()
        return self._tok, self._model

    @torch.inference_mode()
    def encode(self, texts, lang: str = "eng_Latn", batch_size: int = 64,
               normalize: bool | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)
        tok, model = self._load()
        rows = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=_MAX_LEN, return_tensors="pt").to(self.device)
            cls = model(**enc).last_hidden_state[:, 0].float()
            rows.append(torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy())
        return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
