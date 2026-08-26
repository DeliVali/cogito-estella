"""Loop de entrenamiento del ConceptTransformer.

v0.2.0: objetivo MSE (predecir el siguiente concepto) como validación de ingeniería
del backbone y el pipeline de datos (overfit-test). El objetivo real (CE propagada
por el decoder SONAR congelado) llega en v0.2.1. Checkpoints reanudables.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def build_sequences(dataset, seq_len: int) -> np.ndarray:
    """Agrupa conceptos por doc_id (en orden) y los parte en chunks de seq_len.
    No cruza fronteras de documento. Devuelve [n_seq, seq_len, 1024] float32.
    """
    groups: dict[str, list[np.ndarray]] = {}
    order: list[str] = []
    for i in range(len(dataset)):
        emb, meta = dataset[i]
        doc = meta["doc_id"]
        if doc not in groups:
            groups[doc] = []
            order.append(doc)
        groups[doc].append(np.asarray(emb, dtype=np.float32))
    seqs = []
    for doc in order:
        embs = groups[doc]
        n_chunks = len(embs) // seq_len
        for c in range(n_chunks):
            seqs.append(np.stack(embs[c * seq_len:(c + 1) * seq_len]))
    if not seqs:
        return np.zeros((0, seq_len, 1024), dtype=np.float32)
    return np.stack(seqs).astype(np.float32)


def next_concept_mse(model, batch: torch.Tensor) -> torch.Tensor:
    """batch: [B, T, 1024]. Predice el concepto t+1 a partir de <=t."""
    pred = model(batch)  # [B, T, 1024]
    return F.mse_loss(pred[:, :-1], batch[:, 1:])


def save_checkpoint(path, model, optimizer, step: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step}, path)


def load_checkpoint(path, model, optimizer=None) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    return ck["step"]


def _cosine_lr(step: int, total: int, base_lr: float, warmup: int = 0) -> float:
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def train_loop(model, sequences: np.ndarray, steps: int, lr: float, batch_size: int,
               device: str = "cpu", loss: str = "mse", out_dir: str | None = None,
               resume: bool = False, log_every: int = 50, ckpt_every: int = 500,
               seed: int = 0) -> list[dict]:
    if loss != "mse":
        raise NotImplementedError("v0.2.0 solo implementa MSE; CE propagada llega en v0.2.1")
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    start_step = 0
    if resume and out_dir and (Path(out_dir) / "last.pt").exists():
        start_step = load_checkpoint(Path(out_dir) / "last.pt", model, opt)

    data = torch.from_numpy(sequences).to(device)
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    metrics = []
    use_amp = device == "cuda"

    for step in range(start_step, steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        batch = data[idx]
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, steps, lr)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = next_concept_mse(model, batch)
            l.backward()
        else:
            l = next_concept_mse(model, batch)
            l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            metrics.append({"step": step, "loss": float(l.item()),
                            "lr": opt.param_groups[0]["lr"], "t": time.time()})
        if out_dir and ckpt_every and step > 0 and step % ckpt_every == 0:
            save_checkpoint(Path(out_dir) / "last.pt", model, opt, step)

    if out_dir:
        save_checkpoint(Path(out_dir) / "last.pt", model, opt, steps)
        with (Path(out_dir) / "metrics.jsonl").open("w") as fh:
            for m in metrics:
                fh.write(json.dumps(m) + "\n")
    return metrics
