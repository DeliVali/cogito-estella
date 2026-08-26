"""Almacén de conceptos: shards de embeddings (fp16) + texto crudo + metadatos.

Formato por shard (un subdirectorio `shard_00000/`, etc.):
- embeddings.npy : float16, forma [n, 1024]
- texts.jsonl    : una línea JSON {"text","lang","source","doc_id"} por fila
- meta.json      : {"n", "sonar_version", "timestamp", "commit"}

El texto crudo se guarda junto al embedding porque el objetivo de entrenamiento
(cross-entropy propagada, estilo SONAR-LLM) necesita los tokens verdaderos.
"""
import json
from pathlib import Path

import numpy as np

EMB_DIM = 1024


class ShardWriter:
    def __init__(self, out_dir: str, shard_size: int = 100_000, resume: bool = False):
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        # reanudar: continuar la numeración desde los shards ya escritos
        self._shard_idx = 0
        if resume:
            existing = [p for p in self.root.iterdir()
                        if p.is_dir() and (p / "embeddings.npy").exists()]
            self._shard_idx = len(existing)
        self._embs: list[np.ndarray] = []
        self._metas: list[dict] = []

    def add(self, embedding: np.ndarray, text: str, lang: str, source: str, doc_id: str) -> None:
        self._embs.append(np.asarray(embedding, dtype=np.float16).reshape(EMB_DIM))
        self._metas.append({"text": text, "lang": lang, "source": source, "doc_id": doc_id})
        if len(self._embs) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._embs:
            return
        shard_dir = self.root / f"shard_{self._shard_idx:05d}"
        shard_dir.mkdir(exist_ok=True)
        np.save(shard_dir / "embeddings.npy", np.stack(self._embs))
        with (shard_dir / "texts.jsonl").open("w", encoding="utf-8") as fh:
            for meta in self._metas:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        (shard_dir / "meta.json").write_text(json.dumps({"n": len(self._embs)}))
        self._shard_idx += 1
        self._embs = []
        self._metas = []

    def close(self) -> None:
        self._flush()


class ConceptDataset:
    def __init__(self, root: str):
        self.root = Path(root)
        self._shards: list[Path] = sorted(
            p for p in self.root.iterdir() if p.is_dir() and (p / "embeddings.npy").exists()
        ) if self.root.exists() else []
        self._index: list[tuple[int, int]] = []  # (shard_i, row_i)
        self._embs_cache: dict[int, np.ndarray] = {}
        self._metas: list[list[dict]] = []
        for si, shard in enumerate(self._shards):
            metas = [json.loads(ln) for ln in (shard / "texts.jsonl").read_text(encoding="utf-8").splitlines()]
            self._metas.append(metas)
            for ri in range(len(metas)):
                self._index.append((si, ri))

    def __len__(self) -> int:
        return len(self._index)

    def _embs(self, shard_i: int) -> np.ndarray:
        if shard_i not in self._embs_cache:
            self._embs_cache[shard_i] = np.load(self._shards[shard_i] / "embeddings.npy", mmap_mode="r")
        return self._embs_cache[shard_i]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        shard_i, row_i = self._index[idx]
        emb = np.asarray(self._embs(shard_i)[row_i], dtype=np.float16)
        return emb, self._metas[shard_i][row_i]
