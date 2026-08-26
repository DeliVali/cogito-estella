"""Loop de entrenamiento del GraphDecoder a escala, con checkpoints reanudables y
registro de la curva de aprendizaje (para la fase de producción v0.5.0).

Objetivo de grafo (barato, no-autorregresivo): pérdida = BCE existencia + CE etiquetas
+ BCE adyacencia. Checkpoints cada `ckpt_every` pasos (config: 5% del total) para
monitorear el aprendizaje y reanudar ante fallos.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cogito_estella.model.graph_decoder import graph_loss
from cogito_estella.model.train import load_checkpoint, save_checkpoint


def _cosine_lr(step, total, base_lr, warmup=0):
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def train_loop_graph(model, emb: np.ndarray, exist: np.ndarray, labels: np.ndarray,
                     adj: np.ndarray, steps: int, lr: float, batch_size: int,
                     device: str = "cpu", out_dir: str | None = None, resume: bool = False,
                     log_every: int = 500, ckpt_every: int = 5000, seed: int = 0,
                     eval_fn=None) -> list[dict]:
    """emb [N,1024]; exist [N,K]; labels [N,K]; adj [N,R,K,K]. eval_fn(model)->dict opcional
    para registrar métricas (p. ej. Triple F1 held-out) en cada checkpoint."""
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    start = 0
    if resume and out_dir and (Path(out_dir) / "last.pt").exists():
        start = load_checkpoint(Path(out_dir) / "last.pt", model, opt)

    E = torch.from_numpy(emb).float().to(device)
    TE = torch.from_numpy(exist).float().to(device)
    TL = torch.from_numpy(labels).long().to(device)
    TA = torch.from_numpy(adj).float().to(device)
    n = E.shape[0]
    rng = np.random.default_rng(seed)
    metrics = []
    mpath = Path(out_dir) / "metrics.jsonl" if out_dir else None

    def checkpoint(step):
        if not out_dir:
            return
        save_checkpoint(Path(out_dir) / "last.pt", model, opt, step)
        row = {"step": step, "loss": float(last_loss), "lr": opt.param_groups[0]["lr"],
               "t": time.time()}
        if eval_fn is not None:
            model.eval()
            with torch.no_grad():
                row.update(eval_fn(model))
            model.train()
        with mpath.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    last_loss = float("nan")
    t0 = time.time()
    for step in range(start, steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, steps, lr)
        out = model(E[idx])
        loss = graph_loss(out, TE[idx], TL[idx], TA[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = loss.item()
        if step % log_every == 0 or step == steps - 1:
            metrics.append({"step": step, "loss": last_loss})
        if out_dir and ckpt_every and step > start and step % ckpt_every == 0:
            checkpoint(step)
        if device == "cuda":
            torch.cuda.empty_cache()
    if out_dir:
        checkpoint(steps)
    return metrics
