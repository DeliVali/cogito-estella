"""Reconstruction-fidelity metrics. No GPU, no network."""
import json
import re
import statistics

from sacrebleu.metrics import CHRF

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_CHRF = CHRF()


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def exact_match(ref: str, hyp: str) -> bool:
    return normalize_ws(ref) == normalize_ws(hyp)


def chrf(ref: str, hyp: str) -> float:
    return _CHRF.sentence_score(hyp, [ref]).score


def number_fidelity(ref: str, hyp: str) -> float | None:
    ref_nums = _NUM_RE.findall(ref)
    if not ref_nums:
        return None
    hyp_nums = _NUM_RE.findall(hyp)
    remaining = list(hyp_nums)
    kept = 0
    for num in ref_nums:
        if num in remaining:
            remaining.remove(num)
            kept += 1
    return kept / len(ref_nums)


def json_valid(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def json_equiv(ref: str, hyp: str) -> bool:
    if not (json_valid(ref) and json_valid(hyp)):
        return False
    return json.loads(ref) == json.loads(hyp)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("empty scores")
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(scores: list[float]) -> dict:
    vals = sorted(scores)
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "p10": _percentile(vals, 0.10),
        "p90": _percentile(vals, 0.90),
    }
