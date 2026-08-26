"""CE loss back-propagated through the frozen SONAR decoder (SONAR-LLM mechanism).

The concept model predicts an embedding ê_t; we feed it as the encoder output to the
**frozen** SONAR decoder with teacher forcing over the target sentence's true tokens,
and compute token-level cross-entropy. The gradient flows through the frozen decoder
back to ê_t, giving a likelihood signal (unlike MSE, which collapses to the mean, or
diffusion, which yields no log-probs).

Empirically validated (exp004): a sentence's true embedding gives CE ~0.3-0.5, a random
one ~20 — the loss discriminates correctly.
"""
import torch
import torch.nn.functional as F


class SonarCELoss:
    """Wraps the frozen SONAR decoder to compute CE with teacher forcing.

    Decoder and tokenizer stay frozen (requires_grad_(False), eval).
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
        self.max_tokens = max_tokens  # length cap: bounds logits memory (256k vocab)
        self.model = pipe.model.eval().requires_grad_(False)
        self.tokenizer = pipe.tokenizer
        self.pad_idx = self.tokenizer.vocab_info.pad_idx
        self.eos_idx = self.tokenizer.vocab_info.eos_idx
        self._encoders: dict[str, object] = {}
        # decoder param dtype (to cast the input embedding); loading in bf16 halves
        # the 256k-vocab logits memory.
        self._param_dtype = next(self.model.parameters()).dtype

    def _encoder(self, lang: str):
        if lang not in self._encoders:
            self._encoders[lang] = self.tokenizer.create_encoder(
                task="translation", lang=lang, mode="target")
        return self._encoders[lang]

    def tokenize(self, texts: list[str], lang: str) -> tuple[torch.Tensor, torch.Tensor]:
        """texts -> (tgt_in, labels) with teacher forcing and padding. Both [B, T-1]."""
        enc = self._encoder(lang)
        id_seqs = []
        for t in texts:
            ids = enc(t)
            if ids.shape[0] > self.max_tokens:  # truncate and force a trailing EOS
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
        """pred_emb: [B, 1024] (with grad). Return scalar CE averaged over non-pad tokens.

        The gradient flows to pred_emb; the decoder stays frozen.
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
        """Like loss() but returns (CE_sum, n_non_pad_tokens) to aggregate correctly
        across batches (token-weighted mean, batch-invariant)."""
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
