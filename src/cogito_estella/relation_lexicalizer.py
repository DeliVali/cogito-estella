"""Lexical relation labels from the dependency path between two entity spans.
Pure functions over a spaCy Doc; no torch. Pattern ids: 1 passive agent, 2 active verb,
3 verb+prep, 4 passive/reduced-relative+prep, 5 copula is_a, 6 nominal prep, 7 compound."""
from __future__ import annotations

import re
from dataclasses import dataclass

SUBJ = ("nsubj",)
PASS = ("nsubjpass",)
OBJ = ("dobj", "obj")
_CLEAN = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True)
class Lex:
    label: str
    pattern: int
    swapped: bool


def _clean(label: str) -> str:
    return _CLEAN.sub("_", label.lower()).strip("_")


def head_token(doc, span):
    """Root token of the char span, or None when the span lies outside the doc."""
    start, end = span
    if start < 0 or end > len(doc.text) or start >= end:
        return None
    sp = doc.char_span(start, end, alignment_mode="expand")
    return None if sp is None else sp.root


def verb_label(verb, prep: str | None = None) -> str:
    """lemma[_particle][_prep], `not_` prefix when the verb carries a `neg` child."""
    label = verb.lemma_.lower()
    for c in verb.children:
        if c.dep_ == "prt":
            label += "_" + c.text.lower()
    if prep:
        label += "_" + prep.lower()
    if any(c.dep_ == "neg" for c in verb.children):
        label = "not_" + label
    return _clean(label)


def _prep_of(q):
    """(governor, preposition) when q is the object of a preposition, else None."""
    if q.dep_ == "pobj" and q.head.dep_ == "prep":
        return q.head.head, q.head.lemma_.lower()
    return None


def _agent_of(q):
    """Verb governing q when q is the agent of a passive (`by` phrase), else None."""
    if q.dep_ == "pobj" and q.head.dep_ == "agent":
        return q.head.head
    return None


def _passive_verb(p):
    """Verb of which p is the patient: nsubjpass, or head of a reduced relative (acl)."""
    if p.dep_ in PASS:
        return p.head
    for c in p.children:
        if c.dep_ == "acl" and c.pos_ == "VERB" and c.tag_ == "VBN":
            return c
    return None


def _match(p, q) -> tuple[str, int] | None:
    """(label, pattern) with p as printed subject and q as printed object, else None."""
    # row 1: q is the passive patient, p its agent -> agent acts on patient
    v = _agent_of(p)
    if v is not None and q.dep_ in PASS and q.head == v:
        return verb_label(v), 1
    # row 2: nsubj + dobj under the same verb
    if p.dep_ in SUBJ and q.dep_ in OBJ and q.head == p.head:
        return verb_label(p.head), 2
    # row 3: nsubj + prep object of an active verb
    pq = _prep_of(q)
    if p.dep_ in SUBJ and pq is not None and pq[0] == p.head \
            and p.head.pos_ in ("VERB", "AUX"):
        return verb_label(p.head, pq[1]), 3
    # row 4: passive patient + prep object of the same verb (no agent)
    v = _passive_verb(p)
    if v is not None and pq is not None and pq[0] == v:
        return verb_label(v, pq[1]), 4
    # row 5: copula with attr predicate (is_a)
    if p.dep_ in SUBJ and q.dep_ == "attr" and q.head == p.head \
            and p.head.lemma_.lower() == "be":
        return "is_a", 5
    # row 6: nominal possessive (prep object of a noun)
    if p.pos_ in ("NOUN", "PROPN") and pq is not None and pq[0] == p:
        return _clean(pq[1]), 6
    # row 7: compound modifier
    if q.dep_ == "compound" and q.head == p:
        return "compound", 7
    return None


def lexicalize(doc, s_span, o_span) -> Lex | None:
    a, b = head_token(doc, s_span), head_token(doc, o_span)
    if a is None or b is None or a == b or a.sent.start != b.sent.start:
        return None
    for p, q, swapped in ((a, b, False), (b, a, True)):
        hit = _match(p, q)
        if hit:
            return Lex(hit[0], hit[1], swapped)
    return None


_STOP = {"the", "a", "an", "this", "that", "these", "those", "its", "their", "our", "his",
         "her", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
         "will", "would", "can", "could", "may", "might", "should", "do", "does", "did",
         "often", "also", "then", "and", "or", "quickly", "reliably", "very", "which",
         "who", "it", "we", "they", "there"}
_WORD = re.compile(r"[A-Za-z0-9\-']+")


def between_spans(sentence: str, s_span, o_span) -> Lex | None:
    """Control B: up to three content tokens strictly between the spans; `by` swaps."""
    (_a0, a1), (b0, _b1) = sorted((tuple(s_span), tuple(o_span)))
    words = [w.lower() for w in _WORD.findall(sentence[a1:b0])]
    swapped = "by" in words
    keep = [w for w in words if w not in _STOP][:3]
    if not keep:
        return None
    if tuple(s_span) > tuple(o_span):        # spans given in reverse text order
        swapped = not swapped
    return Lex(_clean("_".join(keep)), 0, swapped)
