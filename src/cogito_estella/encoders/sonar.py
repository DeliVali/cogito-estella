"""SONAR adapter. Weights are CC-BY-NC 4.0; ships in the [sonar] extra."""
from __future__ import annotations

import numpy as np
import torch

_CARD = "text_sonar_basic_encoder"
_DIM = 1024


class SonarEncoder:
    """`TextToEmbeddingModelPipeline` behind the TextEncoder contract.

    Native output is raw: the published heads were trained on unnormalized SONAR
    vectors, so `normalize=None` must not rescale.
    """

    name = "sonar"
    revision = _CARD
    dim = _DIM
    native_normalized = False

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            self._pipe = TextToEmbeddingModelPipeline(
                encoder=_CARD, tokenizer=_CARD, device=torch.device(self.device))
        return self._pipe

    def encode(self, texts, lang: str = "eng_Latn", batch_size: int = 64,
               normalize: bool | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)
        # batch_size is explicit: fairseq2 defaults to 5.
        emb = self.pipe.predict(texts, source_lang=lang, batch_size=batch_size)
        mat = emb.detach().to(torch.float32).cpu().numpy()
        if normalize:
            mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
        return np.ascontiguousarray(mat, dtype=np.float32)
