"""Segmentación de texto en unidades tipo oración con SaT (wtpsplit).

SaT (Segment any Text, Minixhofer et al. 2024) es robusto a estilo y corrupción,
a diferencia del split por regex. Modelo por defecto: sat-3l-sm (mejor tradeoff
velocidad/calidad para GPU de 12 GB, per README de wtpsplit).
"""
from cogito_estella.sampling import good_unit


class Segmenter:
    def __init__(self, model: str = "sat-3l-sm", device: str | None = None,
                 threshold: float | None = None):
        import torch
        from wtpsplit import SaT

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold
        self._sat = SaT(model)
        if device == "cuda":
            self._sat.half().to(device)
        else:
            self._sat.to(device)

    def _clean(self, units: list[str]) -> list[str]:
        out = []
        for u in units:
            u = u.strip()
            if u and good_unit(u):
                out.append(u)
        return out

    def segment(self, text: str) -> list[str]:
        kwargs = {"threshold": self.threshold} if self.threshold is not None else {}
        return self._clean(self._sat.split(text, **kwargs))

    def segment_batch(self, texts: list[str]) -> list[list[str]]:
        kwargs = {"threshold": self.threshold} if self.threshold is not None else {}
        return [self._clean(units) for units in self._sat.split(texts, **kwargs)]
