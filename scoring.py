
"""
Nirmaan AI – Spoken Introduction Scorer (NLP + Rule-based Hybrid)
================================================================

This module implements a *product-style* scoring engine for the Nirmaan
spoken introduction case study.

As an ML engineer, the design choices are:

- Keep the scoring **transparent & explainable** (rubric-style rules)
- Add **NLP models** for semantic quality & sentiment / engagement
- Return a clean JSON structure that a backend / dashboard can consume

Dependencies (installed via requirements.txt):
- sentence-transformers
- transformers
- torch
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
from collections import Counter
import math

import numpy as np

# Lazy-import heavy NLP models so that unit tests / simple scripts stay fast
_nlp_models = {
    "loaded": False,
    "sentiment": None,
    "embedder": None,
}


def _load_nlp_models():
    """
    Load Hugging Face models only once.

    - Sentiment: distilbert-base-uncased-finetuned-sst-2-english
    - Sentence embeddings: all-MiniLM-L6-v2
    """
    global _nlp_models
    if _nlp_models["loaded"]:
        return _nlp_models

    from transformers import pipeline
    from sentence_transformers import SentenceTransformer

    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    _nlp_models["sentiment"] = sentiment
    _nlp_models["embedder"] = embedder
    _nlp_models["loaded"] = True
    return _nlp_models


WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


# ---------------- Rule-based rubric pieces ----------------

SALUTATION_KEYWORDS = {
    "excellent": ["i am excited", "it is my pleasure", "i am very happy", "i am glad"],
    "good": ["good morning", "good afternoon", "good evening"],
    "normal": ["hi ", "hello ", "hey "],
}


MUST_HAVE_SLOTS = {
    "name": ["my name is", "i am ", "i'm "],
    "class_or_grade": ["class ", "grade ", "standard ", "i study in", "i am studying in"],
    "school": ["school", "college", "university"],
    "hobbies_or_goals": [
        "my hobby", "my hobbies", "my interests",
        "i like to", "i love to",
        "my dream is", "my goal is", "my aim is",
        "i want to become", "i want to be",
    ],
}

GOOD_TO_HAVE_SLOTS = {
    "age": ["years old", "year old"],
    "family": ["my family", "my father", "my mother", "my parents", "siblings"],
    "origin": ["i am from", "i'm from", "i belong to", "i come from"],
    "thanks": ["thank you", "thanks for listening", "that's all about me"],
}


FILLER_WORDS = [
    "um", "uh", "er", "ah", "like", "you know", "kind of", "sort of",
    "basically", "actually", "literally", "so yeah",
]


@dataclass
class CriterionScore:
    name: str
    max_score: float
    score: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["score_normalized_0_1"] = round(self.score / self.max_score, 3) if self.max_score else None
        return obj


@dataclass
class EvaluationResult:
    overall_score: float
    max_score: float
    criteria: List[CriterionScore]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "max_score": self.max_score,
            "criteria": [c.to_dict() for c in self.criteria],
            "meta": self.meta,
        }


def score_salutation(text: str) -> CriterionScore:
    txt = text.lower() + " "
    tier = "none"
    for key, phrases in SALUTATION_KEYWORDS.items():
        if any(p in txt for p in phrases):
            tier = key
            break
    mapping = {"none": 0, "normal": 2, "good": 4, "excellent": 5}
    score = mapping[tier]
    return CriterionScore(
        name="Salutation quality (rule-based)",
        max_score=5,
        score=score,
        details={"tier": tier},
    )


def _slot_present(text_lower: str, patterns: List[str]) -> bool:
    return any(p in text_lower for p in patterns)


def score_content_slots(text: str) -> List[CriterionScore]:
    txt = text.lower()

    must_hits, must_score = {}, 0
    for slot, patterns in MUST_HAVE_SLOTS.items():
        present = _slot_present(txt, patterns)
        must_hits[slot] = present
        if present:
            must_score += 5  # each must-have → 5 points (max 20)

    good_hits, good_score = {}, 0
    for slot, patterns in GOOD_TO_HAVE_SLOTS.items():
        present = _slot_present(txt, patterns)
        good_hits[slot] = present
        if present:
            good_score += 2  # each good-to-have → 2 points

    crits = [
        CriterionScore(
            name="Core content coverage (must-have)",
            max_score=5 * len(MUST_HAVE_SLOTS),
            score=must_score,
            details={"slots": must_hits},
        ),
        CriterionScore(
            name="Additional details (good-to-have)",
            max_score=2 * len(GOOD_TO_HAVE_SLOTS),
            score=good_score,
            details={"slots": good_hits},
        ),
    ]
    return crits


def score_filler_and_length(tokens: List[str]) -> List[CriterionScore]:
    n_tokens = len(tokens)
    if n_tokens == 0:
        return [
            CriterionScore("Clarity (filler words)", 10, 0, {"note": "empty transcript"}),
            CriterionScore("Length appropriateness", 10, 0, {"note": "empty transcript"}),
        ]

    # Filler rate
    token_str = " ".join(tokens)
    filler_count = 0
    for fw in FILLER_WORDS:
        if " " in fw:
            filler_count += token_str.count(fw)
        else:
            filler_count += tokens.count(fw)

    filler_per_100 = filler_count * 100.0 / max(1, n_tokens)
    if filler_per_100 <= 3:
        filler_score = 10
    elif filler_per_100 <= 6:
        filler_score = 8
    elif filler_per_100 <= 10:
        filler_score = 6
    else:
        filler_score = 3

    # Length (we expect around 40–120 tokens)
    if n_tokens < 30:
        length_score = 4
    elif n_tokens <= 120:
        length_score = 10
    elif n_tokens <= 200:
        length_score = 8
    else:
        length_score = 5

    return [
        CriterionScore(
            name="Clarity (filler words per 100 tokens)",
            max_score=10,
            score=filler_score,
            details={
                "filler_count": filler_count,
                "tokens": n_tokens,
                "filler_per_100": round(filler_per_100, 2),
            },
        ),
        CriterionScore(
            name="Length appropriateness",
            max_score=10,
            score=length_score,
            details={
                "tokens": n_tokens,
                "note": "ideal range ≈ 40–120 tokens",
            },
        ),
    ]


def score_ttr(tokens: List[str]) -> CriterionScore:
    if not tokens:
        return CriterionScore(
            name="Vocabulary richness (TTR)",
            max_score=10,
            score=0,
            details={"ttr": 0.0},
        )
    unique = set(tokens)
    ttr = len(unique) / len(tokens)
    if ttr >= 0.9:
        score = 10
    elif ttr >= 0.7:
        score = 8
    elif ttr >= 0.5:
        score = 6
    elif ttr >= 0.3:
        score = 4
    else:
        score = 2

    return CriterionScore(
        name="Vocabulary richness (type–token ratio)",
        max_score=10,
        score=score,
        details={"ttr": round(ttr, 3)},
    )


def score_speech_rate(tokens: List[str], duration_seconds: Optional[float]) -> CriterionScore:
    if not duration_seconds or duration_seconds <= 0:
        return CriterionScore(
            name="Speech rate (WPM – optional)",
            max_score=10,
            score=0,
            details={
                "wpm": None,
                "note": "duration_seconds not provided; not used in final score",
                "not_applicable": True,
            },
        )
    wpm = len(tokens) / (duration_seconds / 60.0)
    if wpm < 80:
        score = 4
    elif wpm <= 140:
        score = 10  # ideal
    elif wpm <= 170:
        score = 7
    else:
        score = 4

    return CriterionScore(
        name="Speech rate (words per minute)",
        max_score=10,
        score=score,
        details={"wpm": round(wpm, 2)},
    )


# ---------------- NLP-based scoring ----------------

def semantic_quality_score(text: str) -> CriterionScore:
    """
    Use sentence-transformers to compute similarity to an 'ideal intro' description.
    """
    models = _load_nlp_models()
    embedder = models["embedder"]
    ideal_prompt = (
        "A clear, confident self introduction by a student, including name, "
        "class or age, school, family background, hobbies, and future goals, "
        "spoken in 30 to 60 seconds."
    )
    emb = embedder.encode([ideal_prompt, text], normalize_embeddings=True)
    ideal_vec, text_vec = emb[0], emb[1]
    sim = float(np.dot(ideal_vec, text_vec))  # cosine because normalized
    # map [-1,1] to [0,1] then to score 0–20
    sim01 = max(0.0, (sim + 1.0) / 2.0)
    score = sim01 * 20.0

    return CriterionScore(
        name="Semantic coverage vs ideal introduction (NLP)",
        max_score=20,
        score=score,
        details={
            "cosine_similarity": round(sim, 3),
            "normalized_0_1": round(sim01, 3),
        },
    )


def sentiment_engagement_score(text: str) -> CriterionScore:
    """
    Use a sentiment model as a proxy for enthusiasm / engagement.
    """
    models = _load_nlp_models()
    sentiment_pipe = models["sentiment"]
    result = sentiment_pipe(text[:512])[0]  # truncate very long text
    label = result["label"]
    score_val = float(result["score"])

    if label.upper().startswith("NEG"):
        positivity = 1.0 - score_val
    else:
        positivity = score_val

    engagement_score = positivity * 20.0  # 0–20

    return CriterionScore(
        name="Engagement / positivity (NLP)",
        max_score=20,
        score=engagement_score,
        details={
            "raw_label": label,
            "raw_score": round(score_val, 3),
            "positivity_index_0_1": round(positivity, 3),
        },
    )


# ---------------- Main public API ----------------

def evaluate_transcript(
    transcript: str,
    duration_seconds: Optional[float] = None,
) -> EvaluationResult:
    """
    Evaluate a student's spoken introduction transcript.

    Returns:
        EvaluationResult object (with `.to_dict()` helper)
    """
    text = transcript.strip()
    tokens = tokenize(text)

    criteria: List[CriterionScore] = []

    # Rule-based rubric portion
    criteria.append(score_salutation(text))
    criteria.extend(score_content_slots(text))
    criteria.extend(score_filler_and_length(tokens))
    criteria.append(score_ttr(tokens))
    criteria.append(score_speech_rate(tokens, duration_seconds))

    # NLP-driven portion
    if text:
        criteria.append(semantic_quality_score(text))
        criteria.append(sentiment_engagement_score(text))

    # Aggregate
    raw_total = 0.0
    raw_max = 0.0
    for c in criteria:
        if c.details.get("not_applicable"):
            # don't add to max if metric couldn't be computed
            continue
        raw_total += c.score
        raw_max += c.max_score

    overall = 0.0 if raw_max == 0 else raw_total / raw_max * 100.0

    meta = {
        "word_count": len(tokens),
        "duration_seconds": duration_seconds,
        "raw_total": raw_total,
        "raw_max": raw_max,
        "uses_nlp_models": True,
    }

    return EvaluationResult(
        overall_score=overall,
        max_score=100.0,
        criteria=criteria,
        meta=meta,
    )


if __name__ == "__main__":
    sample = (
        "Good morning, my name is Aparna. I am 13 years old and I study in class 8. "
        "I am from Hyderabad and I live with my family. My hobbies are reading and playing badminton. "
        "My goal is to become a scientist. Thank you for listening."
    )
    res = evaluate_transcript(sample, duration_seconds=60)
    import json
    print(json.dumps(res.to_dict(), indent=2))
