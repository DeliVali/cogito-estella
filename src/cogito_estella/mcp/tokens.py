"""Token accounting: proxy tokenizer and the ledger of what the agent actually receives."""
from __future__ import annotations

from collections import defaultdict

_ENC = None
_TRIED = False


def ntok(text: str) -> int:
    """cl100k_base token count when tiktoken is available, else len//4 (never 0)."""
    global _ENC, _TRIED
    if not _TRIED:
        _TRIED = True
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except (ImportError, OSError, ValueError):      # missing package, download blocked, or hash mismatch
            _ENC = None
    if _ENC is None:
        return max(1, len(text) // 4)
    return max(1, len(_ENC.encode(text, disallowed_special=())))


class Ledger:
    """Raw tokens ingested vs tokens served to the agent, by tool."""

    def __init__(self) -> None:
        self.raw_tokens = 0
        self.served = 0
        self.calls: dict[str, int] = defaultdict(int)
        self.served_by: dict[str, int] = defaultdict(int)

    def charge(self, tool: str, payload: str) -> int:
        n = ntok(payload)
        self.served += n
        self.calls[tool] += 1
        self.served_by[tool] += n
        return n
