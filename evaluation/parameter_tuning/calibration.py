"""Calibration gate: prove the matrix scores reproduce the live reranker.

Runs after the cache build, BEFORE the search. If the recomputed matrix
scores with the current settings weights do not match the live reranked
scores bit-for-bit (up to float tolerance), tuning would optimize a score
function that production never computes - so we abort here.
"""

import numpy as np

from app.config.settings import get_settings

from .cache import CaseCache
from .matrices import FeatureMatrices

TOLERANCE = 1e-6

def calibrate(cases: list[CaseCache], matrices: FeatureMatrices,
              prompt_names: list[str], n_samples: int = 15) -> dict:
    """Compare matrix scores against live reranked scores on a sample."""
    settings = get_settings()
    if len(settings.DECISION_MODEL_WEIGHTS) != len(prompt_names):
        raise ValueError(
            "DECISION_MODEL_WEIGHTS must contain one weight for each "
            "decision_model_prompts entry."
        )

    norm_key = int(settings.KEYWORD_MATCH_NORMALIZATION)
    if norm_key not in matrices.keyword:
        raise ValueError(
            f"KEYWORD_MATCH_NORMALIZATION={norm_key} was not cached; add it "
            "to NORMALIZATION_CANDIDATES"
        )

    kwargs = dict(
        intent_weight=settings.INTENT_WEIGHT,
        bonus=settings.INTENT_MATCH_BONUS,
        penalty=settings.INTENT_MISMATCH_PENALTY,
        kw_weight=settings.KEYWORD_WEIGHT,
        dm_weights=list(settings.DECISION_MODEL_WEIGHTS),
        normalization=norm_key,
    )

    step = max(1, len(cases) // n_samples)
    sample = list(range(0, len(cases), step))[:n_samples]
    results = []

    for i in sample:
        case = cases[i]
        scores = matrices.scores(**kwargs)[i]
        diff = float(np.max(np.abs(scores - case.reranked_scores)))
        order = np.argsort(-scores, kind="stable")
        predicted = case.ids[order[0]].partition(":")[0]
        results.append({
            "question": case.question,
            "max_abs_diff": diff,
            "scores_match": diff <= TOLERANCE,
            "top_match": predicted == case.reranked_top_id,
            "predicted": predicted,
            "live_top": case.reranked_top_id,
        })

    failed = [r for r in results if not (r["scores_match"] and r["top_match"])]
    return {"passed": not failed, "checks": results}