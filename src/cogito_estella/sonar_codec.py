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

    def roundtrip(self, texts: list[str], lang: str, batch_size: int = 32) -> list[str]:
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self._enc.predict(batch, source_lang=lang, batch_size=batch_size)
            dec = self._dec.predict(emb, target_lang=lang, batch_size=batch_size, max_seq_len=512)
            out.extend(dec)
        return out
