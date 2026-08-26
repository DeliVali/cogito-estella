"""Wrapper encode/decode del espacio SONAR. Carga perezosa; device auto."""
import torch


class SonarCodec:
    def __init__(self, device: str | None = None):
        from sonar.inference_pipelines.text import (
            EmbeddingToTextModelPipeline,
            TextToEmbeddingModelPipeline,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._enc = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
        )
        self._dec = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
        )

    def encode(self, texts: list[str], lang: str, batch_size: int = 32):
        """Texto -> embeddings SONAR. Devuelve un tensor [N, 1024]."""
        return self._enc.predict(texts, source_lang=lang, batch_size=batch_size)

    def decode(self, embeddings, lang: str, batch_size: int = 32, max_seq_len: int = 512) -> list[str]:
        """Embeddings SONAR -> texto. Resiliente a OOM: ante CUDA OOM parte el
        batch a la mitad recursivamente (unidades largas de código pueden agotar
        la VRAM del decoder en GPUs de 12 GB)."""
        import torch

        n = embeddings.shape[0] if hasattr(embeddings, "shape") else len(embeddings)
        try:
            return self._dec.predict(embeddings, target_lang=lang, batch_size=batch_size,
                                     max_seq_len=max_seq_len)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if n <= 1:
                # una sola unidad no cabe ni recortando el batch: recorta la longitud
                return self._dec.predict(embeddings, target_lang=lang, batch_size=1,
                                         max_seq_len=min(max_seq_len, 128))
            half = n // 2
            left = self.decode(embeddings[:half], lang, batch_size=max(1, batch_size // 2),
                               max_seq_len=max_seq_len)
            right = self.decode(embeddings[half:], lang, batch_size=max(1, batch_size // 2),
                                max_seq_len=max_seq_len)
            return left + right

    def roundtrip(self, texts: list[str], lang: str, batch_size: int = 16) -> list[str]:
        import torch

        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.encode(batch, lang, batch_size)
            out.extend(self.decode(emb, lang, batch_size))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return out
