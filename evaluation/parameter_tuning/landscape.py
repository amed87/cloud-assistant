"""Phase 3: analytic answerability-gate landscape.

The gate accepts iff top_score >= min_score and gap >= min_gap, so for a
fixed reranker configuration the full confusion matrix over ALL threshold
pairs follows from per-case (top, gap, correct) triples alone.

Threshold candidates are MIDPOINTS between consecutive observed values, not
the observed values themselves: the decision model's probabilities jitter by
roughly ±0.001-0.004 between processes (CPU thread scheduling changes
floating-point summation order), so a threshold sitting exactly on an
observed score is a knife edge - one borderline case flips accept/reject and
moves the whole F0.5. Midpoints keep half the local gap of slack on both
sides of the boundary.
"""

from dataclasses import dataclass

import numpy as np

from app.quality.answerability.answerability_gate import AnswerabilityGate
from app.vectorstores.models import SearchResult

from .cache import CaseCache


@dataclass
class Landscape:
    """Best threshold pair plus the full F0.5 grid over threshold candidates."""

    best_f05: float
    min_score: float
    min_gap: float
    best_counts: dict[str, int]
    score_thresholds: np.ndarray   # (S,)
    gap_thresholds: np.ndarray     # (G,)
    f05_grid: np.ndarray           # (S, G)


def confusion_f05(tp, fp, fn, wrong):
    """F0.5 with wrong accepted entries counted as false positives."""
    p_den = tp + fp + wrong
    r_den = tp + fn + wrong
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(p_den > 0, tp / p_den, 0.0)
        r = np.where(r_den > 0, tp / r_den, 0.0)
        denom = 0.25 * p + r
        return np.where(denom > 0, 1.25 * p * r / denom, 0.0)


def midpoint_thresholds(values: np.ndarray) -> np.ndarray:
    """Candidate thresholds half-way between consecutive unique observed values.

    Sentinels keep the extreme configurations reachable: one below the lowest
    observed value (accept everything at or above the smallest score) and one
    above the highest (reject everything).
    """
    u = np.unique(values)
    if len(u) == 1:
        return np.array([u[0] - 1.0, u[0] + 1.0])
    mid = (u[:-1] + u[1:]) / 2.0
    low_pad = u[0] - (u[1] - u[0]) / 2.0
    return np.concatenate([[low_pad], mid, [u[-1] + 1.0]])


def gap_thresholds(values: np.ndarray) -> np.ndarray:
    """Gap candidates: midpoints between unique gaps, plus 0 (= ignore gap)."""
    # 0.0 accepts every non-negative gap, i.e. "no ambiguity criterion".
    return np.unique(np.concatenate([[0.0], midpoint_thresholds(values)]))


def gate_landscape(top, gap, correct, answerable) -> Landscape:
    """Evaluate every (min_score, min_gap) pair analytically."""
    ans0 = np.asarray(answerable, dtype=bool)
    cor0 = np.asarray(correct, dtype=bool)
    s_th = midpoint_thresholds(top)
    g_th = gap_thresholds(gap)

    # acc[c, s, g]: gate accepts case c at threshold pair (s, g)
    acc = (top[:, None, None] >= s_th[None, :, None]) & \
          (gap[:, None, None] >= g_th[None, None, :])
    ans = ans0[:, None, None]
    cor = cor0[:, None, None]

    tp = (acc & ans & cor).sum(axis=0)
    wrong = (acc & ans & ~cor).sum(axis=0)
    fn = (~acc & ans).sum(axis=0)
    fp = (acc & ~ans).sum(axis=0)

    f05 = confusion_f05(tp, fp, fn, wrong)
    s_idx, g_idx = np.unravel_index(int(np.argmax(f05)), f05.shape)

    return Landscape(
        best_f05=float(f05[s_idx, g_idx]),
        min_score=float(s_th[s_idx]),
        min_gap=float(g_th[g_idx]),
        best_counts={
            "tp": int(tp[s_idx, g_idx]), "fp": int(fp[s_idx, g_idx]),
            "fn": int(fn[s_idx, g_idx]), "wrong": int(wrong[s_idx, g_idx]),
        },
        score_thresholds=s_th,
        gap_thresholds=g_th,
        f05_grid=f05,
    )


def counts_at(land: Landscape, min_score: float, min_gap: float,
              top, gap, correct, answerable) -> dict[str, int]:
    """Analytic confusion counts for one arbitrary threshold pair."""
    acc = (np.asarray(top) >= min_score) & (np.asarray(gap) >= min_gap)
    ans = np.asarray(answerable, dtype=bool)
    cor = np.asarray(correct, dtype=bool)
    return {
        "tp": int((acc & ans & cor).sum()),
        "fp": int((acc & ~ans).sum()),
        "fn": int((~acc & ans).sum()),
        "wrong": int((acc & ans & ~cor).sum()),
    }


def robust_f05(land: Landscape, top, gap, correct, answerable, *,
               top_jitter: float, margin_jitter: float,
               repeats: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo F0.5 grid under simulated decision-model score jitter.

    Each repeat perturbs every case's top score by U(-top_jitter, +top_jitter)
    and its gap by U(-margin_jitter, +margin_jitter) - top and runner-up
    scores jitter independently, so the margin gets roughly twice the spread.
    Returns (mean per threshold pair, worst-case per threshold pair), both
    with the same (S, G) shape as ``land.f05_grid``.
    """
    rng = np.random.default_rng(seed)
    s_th = land.score_thresholds
    g_th = land.gap_thresholds
    ans = answerable[:, None, None]
    cor = correct[:, None, None]

    mean = np.zeros_like(land.f05_grid)
    worst = np.full_like(land.f05_grid, np.inf)

    for _ in range(repeats):
        top_p = top + rng.uniform(-top_jitter, top_jitter, size=len(top))
        gap_p = np.maximum(0.0, gap + rng.uniform(-margin_jitter, margin_jitter,
                                                  size=len(gap)))
        acc = (top_p[:, None, None] >= s_th[None, :, None]) & \
              (gap_p[:, None, None] >= g_th[None, None, :])

        tp = (acc & ans & cor).sum(axis=0)
        wrong = (acc & ans & ~cor).sum(axis=0)
        fn = (~acc & ans).sum(axis=0)
        fp = (acc & ~ans).sum(axis=0)
        f05 = confusion_f05(tp, fp, fn, wrong)

        mean += f05 / repeats
        np.minimum(worst, f05, out=worst)

    return mean, worst


def validate_against_gate(cases: list[CaseCache], min_score: float, min_gap: float,
                          scores: np.ndarray) -> dict[str, int]:
    """Cross-check the analytic counts by running the real AnswerabilityGate."""
    gate = AnswerabilityGate(min_score=min_score, min_score_gap=min_gap)
    counts = {"tp": 0, "fp": 0, "fn": 0, "wrong": 0, "tn": 0}

    for case, row in zip(cases, scores):
        order = np.argsort(-row, kind="stable")
        ranked = [
            SearchResult(id=case.ids[i], content=case.contents[i],
                         score=float(row[i]), metadata=case.metadata[i])
            for i in order
        ]
        decision = gate.evaluate_quick(case.question, ranked)
        predicted = decision.document.id.partition(":")[0] if decision.document else None

        if not case.expected:
            counts["fp" if decision.accepted else "tn"] += 1
        elif not decision.accepted:
            counts["fn"] += 1
        elif predicted == case.expected:
            counts["tp"] += 1
        else:
            counts["wrong"] += 1

    return counts