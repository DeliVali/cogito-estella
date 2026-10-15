"""`--multilingual`: the lite index plus sentence vectors from a small multilingual model,
so a question finds its passage in another language. No torch: ONNX Runtime runs the
model, quantized to int8 on this machine the first time (118 MB kept, any CPU).

XQuAD, 1,190 questions per pair, answer in the compact reply at the 300-token default,
lite against lite with vectors: Spanish questions over English text 43 % -> 96 %, Chinese
over English 14 % -> 88 %, English over Chinese 15 % -> 65 %; same language English
95.2 -> 97.8, Spanish 92.0 -> 95.5, Arabic 71.9 -> 82.2, Chinese 76.8 -> 78.7, Thai
62.9 -> 63.4 (p = 0.69)."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from cogito_estella.mcp.store import TextOnlyExtractor

MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
NAME = "multilingual-e5-small-int8"
DIM = 384
MAX_TOKENS = 256
BATCH = 32
INSTALL = "pip install 'cogito-estella[multilingual]'"


def cache_dir() -> Path:
    root = os.environ.get("COGITO_CACHE") or Path(os.environ.get("XDG_CACHE_HOME")
                                                   or Path.home() / ".cache") / "cogito-estella"
    return Path(root) / NAME


def ensure_model(download: bool = True) -> Path:
    """The int8 model and its tokenizer, fetched and quantized once."""
    out = cache_dir()
    if (out / "model.onnx").exists() and (out / "tokenizer.json").exists():
        return out
    if not download:
        raise OSError(f"the multilingual model is not in {out} and downloads are off")
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import QuantType, quantize_dynamic
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:        # the 470 MB float model is not kept
        src = hf_hub_download(MODEL, "onnx/model.onnx", revision=REVISION, local_dir=tmp)
        tok = hf_hub_download(MODEL, "onnx/tokenizer.json", revision=REVISION, local_dir=tmp)
        part = out / "model.onnx.part"
        quantize_dynamic(src, str(part), weight_type=QuantType.QInt8)
        shutil.copy(tok, out / "tokenizer.json")
        part.replace(out / "model.onnx")
    return out


class VectorExtractor(TextOnlyExtractor):
    """Lite sentences plus multilingual-e5 vectors. `ask` interleaves its two rankings
    (store.interleave): the learned lexical order and the vector order."""

    encoder_name = NAME
    dim = DIM
    vectors = True

    def __init__(self, download: bool = True):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(f"--multilingual needs: {INSTALL}") from exc
        self._np = np
        path = ensure_model(download)
        self._tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tok.enable_truncation(MAX_TOKENS)
        self._tok.enable_padding()
        self._sess = ort.InferenceSession(str(path / "model.onnx"),
                                          providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def _encode(self, texts, prefix):
        np = self._np
        out = []
        for i in range(0, len(texts), BATCH):
            enc = self._tok.encode_batch([prefix + t for t in texts[i:i + BATCH]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self._sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / m.sum(1)                 # mean pooling, as e5 was trained
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, DIM), dtype=np.float32)

    def encode_batch(self, texts):
        return self._encode(list(texts), "passage: ")

    def encode_query(self, texts):
        return self._encode(list(texts), "query: ")
