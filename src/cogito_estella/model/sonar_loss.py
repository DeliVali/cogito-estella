"""Pérdida CE propagada por el decoder SONAR congelado (mecanismo de SONAR-LLM).

El modelo de conceptos predice un embedding ê_t; lo alimentamos como salida del
encoder al decoder SONAR **congelado** con teacher forcing sobre los tokens
verdaderos de la oración objetivo, y computamos cross-entropy a nivel token. El
gradiente fluye por el decoder congelado hasta ê_t, dándonos señal de verosimilitud
(a diferencia de MSE, que colapsa al promedio, o difusión, que no da log-probs).

Validado empíricamente (exp004): el embedding verdadero de una oración da CE ~0.3-0.5,
uno aleatorio ~20 — la pérdida discrimina correctamente.
"""
import torch
import torch.nn.functional as F


class SonarCELoss:
    """Envuelve el decoder SONAR congelado para computar CE con teacher forcing.

    El decoder y el tokenizer se mantienen congelados (requires_grad_(False), eval).
    """

    def __init__(self, device: torch.device | str = "cpu", dtype=None, max_tokens: int = 96):
        from fairseq2.nn import BatchLayout
        from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

        self._BatchLayout = BatchLayout
        dev = torch.device(device)
        pipe = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=dev,
            dtype=dtype,
        )
        self.device = dev
        self.max_tokens = max_tokens  # tope de longitud: acota la memoria de logits (vocab 256k)
        self.model = pipe.model.eval().requires_grad_(False)
        self.tokenizer = pipe.tokenizer
        self.pad_idx = self.tokenizer.vocab_info.pad_idx
        self.eos_idx = self.tokenizer.vocab_info.eos_idx
        self._encoders: dict[str, object] = {}
        # dtype de los parámetros del decoder (para castear el embedding de entrada);
        # cargar en bf16 reduce a la mitad la memoria de los logits de vocab 256k.
        self._param_dtype = next(self.model.parameters()).dtype

    def _encoder(self, lang: str):
        if lang not in self._encoders:
            self._encoders[lang] = self.tokenizer.create_encoder(
                task="translation", lang=lang, mode="target")
        return self._encoders[lang]

    def tokenize(self, texts: list[str], lang: str) -> tuple[torch.Tensor, torch.Tensor]:
        """texts -> (tgt_in, labels) con teacher forcing y padding. Ambos [B, T-1]."""
        enc = self._encoder(lang)
        id_seqs = []
        for t in texts:
            ids = enc(t)
            if ids.shape[0] > self.max_tokens:  # truncar y garantizar EOS final
                ids = ids[: self.max_tokens].clone()
                ids[-1] = self.eos_idx
            id_seqs.append(ids)
        maxlen = max(s.shape[0] for s in id_seqs)
        B = len(id_seqs)
        padded = torch.full((B, maxlen), self.pad_idx, dtype=torch.int64)
        for i, s in enumerate(id_seqs):
            padded[i, : s.shape[0]] = s
        tgt_in = padded[:, :-1].to(self.device)
        labels = padded[:, 1:].to(self.device)
        return tgt_in, labels

    def loss(self, pred_emb: torch.Tensor, texts: list[str], lang: str) -> torch.Tensor:
        """pred_emb: [B, 1024] (con grad). Devuelve CE escalar promediada sobre tokens no-pad.

        El gradiente fluye a pred_emb; el decoder queda congelado.
        """
        tgt_in, labels = self.tokenize(texts, lang)
        src = pred_emb.unsqueeze(1).to(device=self.device, dtype=self._param_dtype)  # [B,1,1024]
        src_layout = self._BatchLayout.of(src)
        tgt_layout = self._BatchLayout.of(tgt_in)
        logits = self.model(src, src_layout, tgt_in, tgt_layout)  # [B, 1, T-1, vocab]
        logits = logits[:, 0]  # [B, T-1, vocab]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=self.pad_idx,
        )

    def loss_sum(self, pred_emb: torch.Tensor, texts: list[str], lang: str):
        """Como loss() pero devuelve (suma_CE, n_tokens_no_pad) para agregar de forma
        correcta entre batches (media ponderada por tokens, batch-invariante)."""
        tgt_in, labels = self.tokenize(texts, lang)
        src = pred_emb.unsqueeze(1).to(device=self.device, dtype=self._param_dtype)
        logits = self.model(src, self._BatchLayout.of(src), tgt_in, self._BatchLayout.of(tgt_in))
        logits = logits[:, 0]
        ce_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=self.pad_idx,
            reduction="sum",
        )
        n_tokens = int((labels.reshape(-1) != self.pad_idx).sum().item())
        return ce_sum, n_tokens
