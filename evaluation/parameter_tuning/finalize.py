"""Phase 4: pick the winner, check robustness, export recommended settings."""

from datetime import datetime

import numpy as np
import optuna

from .cache import CaseCache, NORMALIZATION_CANDIDATES
from .landscape import Landscape, counts_at, gate_landscape, robust_f05, validate_against_gate
from .matrices import FeatureMatrices, rank_cases
from .search import DM_RANGE, FLOAT_RANGES

BOUNDARY_EPS = 1e-3      # |value - bound| below this => boundary warning
PLATEAU_TOLERANCE = 0.99 # thresholds within 1% of best F0.5 count as plateau

# Simulated decision-model jitter for robust threshold selection. Observed
# cross-process jitter of the Ollama decision provider's probabilities is
# ~±0.001-0.004 on reranked scores (thread scheduling changes floats
# summation); the margin is a difference of two independently jittering
# scores, hence roughly twice the spread. Values include headroom.
TOP_JITTER = 0.005
MARGIN_JITTER = 0.010
ROBUST_REPEATS = 64
ROBUST_SEED = 0


def finalize(cases: list[CaseCache], matrices: FeatureMatrices,
             prompt_names: list[str], study: optuna.Study) -> dict:
    """Build the full recommendation report from the best trial."""
    best = study.best_trial
    p = best.params

    scores = matrices.scores(
        intent_weight=p["intent_weight"],
        bonus=p["intent_bonus"],
        penalty=p["intent_penalty"],
        kw_weight=p["keyword_weight"],
        dm_weights=[p[f"dm_{i}"] for i in range(len(prompt_names))],
        normalization=p["keyword_normalization"],
    )
    top, gap, predicted = rank_cases(cases, scores)
    correct = np.array([pred == c.expected if c.expected else False
                        for pred, c in zip(predicted, cases)])
    answerable = np.array([bool(c.expected) for c in cases])
    land = gate_landscape(top, gap, correct, answerable)

    # Among all plateau pairs (nominal F0.5 within PLATEAU_TOLERANCE of the
    # best), prefer the one that keeps the best mean F0.5 under simulated
    # decision-model jitter - the exact nominal optimum is usually fragile.
    mean_grid, worst_grid = robust_f05(
        land, top, gap, correct, answerable,
        top_jitter=TOP_JITTER, margin_jitter=MARGIN_JITTER,
        repeats=ROBUST_REPEATS, seed=ROBUST_SEED,
    )
    near = land.f05_grid >= PLATEAU_TOLERANCE * land.best_f05
    r_s_idx, r_g_idx = np.unravel_index(
        int(np.argmax(np.where(near, mean_grid, -np.inf))), mean_grid.shape)
    rec_min_score = float(land.score_thresholds[r_s_idx])
    rec_min_gap = float(land.gap_thresholds[r_g_idx])
    rec_counts = counts_at(land, rec_min_score, rec_min_gap,
                           top, gap, correct, answerable)
    rec_f05 = float(land.f05_grid[r_s_idx, r_g_idx])

    # Gate cross-check: analytic counts vs. the real AnswerabilityGate at the
    # recommended thresholds (not the nominal optimum).
    gate_counts = validate_against_gate(cases, rec_min_score, rec_min_gap, scores)
    validation_match = all(
        gate_counts.get(k, 0) == v
        for k, v in rec_counts.items()
    )

    # Plateau: how flat is the landscape around the optimum?
    s_idx, g_idx = np.where(near)
    plateau = {
        "tolerance": PLATEAU_TOLERANCE,
        "pairs_within_tolerance": int(near.sum()),
        "min_score_range": [float(land.score_thresholds[s_idx].min()),
                            float(land.score_thresholds[s_idx].max())],
        "min_gap_range": [float(land.gap_thresholds[g_idx].min()),
                          float(land.gap_thresholds[g_idx].max())],
    }

    # Boundary warnings: which experience-based ranges did the optimum hit?
    warnings = []
    for name, (lo, hi) in FLOAT_RANGES.items():
        v = p[name]
        if abs(v - lo) < BOUNDARY_EPS or abs(v - hi) < BOUNDARY_EPS:
            warnings.append(f"{name}={v} at range boundary [{lo}, {hi}]")
    for i in range(len(prompt_names)):
        v = p[f"dm_{i}"]
        if abs(v - DM_RANGE[0]) < BOUNDARY_EPS or abs(v - DM_RANGE[1]) < BOUNDARY_EPS:
            warnings.append(f"dm weight '{prompt_names[i]}'={v} at range boundary {DM_RANGE}")
    if p["keyword_normalization"] in (NORMALIZATION_CANDIDATES[0], NORMALIZATION_CANDIDATES[-1]):
        warnings.append(f"keyword_normalization={p['keyword_normalization']} "
                        f"at edge of candidate list")

    return {
        "generated_at": datetime.now().isoformat(),
        "n_cases": len(cases),
        "best_trial": best.number,
        "best_f05": land.best_f05,
        "counts_at_optimum": rec_counts,
        "robustness": {
            "assumed_top_jitter": TOP_JITTER,
            "assumed_margin_jitter": MARGIN_JITTER,
            "repeats": ROBUST_REPEATS,
            "seed": ROBUST_SEED,
            "threshold_selection":
                "plateau pair with best mean F0.5 under simulated jitter",
            "f05_at_recommendation": rec_f05,
            "simulated_mean_f05": float(mean_grid[r_s_idx, r_g_idx]),
            "simulated_worst_f05": float(worst_grid[r_s_idx, r_g_idx]),
        },
        "recommendation": {
            "INTENT_WEIGHT": p["intent_weight"],
            "INTENT_MATCH_BONUS": p["intent_bonus"],
            "INTENT_MISMATCH_PENALTY": p["intent_penalty"],
            "KEYWORD_WEIGHT": p["keyword_weight"],
            "KEYWORD_MATCH_NORMALIZATION": p["keyword_normalization"],
            "DECISION_MODEL_WEIGHTS": [p[f"dm_{i}"] for i in range(len(prompt_names))],
            "RAG_MIN_SCORE": rec_min_score,
            "RAG_MIN_SCORE_GAP": rec_min_gap,
        },
        "plateau": plateau,
        "boundary_warnings": warnings,
        "gate_validation": {"counts": gate_counts, "matches_analytic": validation_match},
    }