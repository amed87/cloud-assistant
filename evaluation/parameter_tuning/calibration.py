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


def resolve_base_weights(cases: list[CaseCache], prompt_names: list[str]) -> dict:
    """Merged, snapshot-first weights used by calibration, search, and finalize.

    With a persisted cache the live scores were produced with the weights
    snapshot stored at cache build time — so all fixed group values
    reconstruct from exactly those weights, independent of the current .env.
    Without a snapshot (fresh retrieval) the current settings weights apply.
    """
    settings = get_settings()
    if len(settings.DECISION_MODEL_WEIGHTS) != len(prompt_names):
        raise ValueError(
            "DECISION_MODEL_WEIGHTS must contain one weight for each "
            "decision_model_prompts entry."
        )

    snapshots = [case.weights_snapshot for case in cases]
    snapshot = snapshots[0] if snapshots and all(s is not None for s in snapshots) else None
    if snapshot is not None and len(snapshot["dm_weights"]) != len(prompt_names):
        raise ValueError(
            "weights snapshot in the cache does not match the current "
            "decision_model_prompts set; rebuild the cache."
        )

    if snapshot is not None:
        return {
            "intent_weight": snapshot["intent_weight"],
            "intent_bonus": snapshot["intent_bonus"],
            "intent_penalty": snapshot["intent_penalty"],
            "kw_weight": snapshot["kw_weight"],
            "dm_weights": list(snapshot["dm_weights"]),
            "normalization": int(snapshot["normalization"]),
            "centroid_facet_weights": list(snapshot["centroid_facet_weights"]),
            "centroid_bonus": snapshot["centroid_bonus"],
            "centroid_penalty": snapshot["centroid_penalty"],
        }
    return {
        "intent_weight": settings.INTENT_WEIGHT,
        "intent_bonus": settings.INTENT_MATCH_BONUS,
        "intent_penalty": settings.INTENT_MISMATCH_PENALTY,
        "kw_weight": settings.KEYWORD_WEIGHT,
        "dm_weights": list(settings.DECISION_MODEL_WEIGHTS),
        "normalization": int(settings.KEYWORD_MATCH_NORMALIZATION),
        "centroid_facet_weights": list(settings.CENTROID_FACET_WEIGHTS),
        "centroid_bonus": settings.CENTROID_MATCH_BONUS,
        "centroid_penalty": settings.CENTROID_MATCH_PENALTY,
    }


def calibrate(cases: list[CaseCache], matrices: FeatureMatrices,
              prompt_names: list[str], n_samples: int = 15) -> dict:
    """Compare matrix scores against live reranked scores on a sample.

    With a persisted cache the live scores were produced with the weights
    snapshot stored at cache build time — so calibration reconstructs with
    exactly those weights, independent of the current .env. Without a
    snapshot (fresh retrieval) the current settings weights apply, which is
    the classic self-consistency check.
    """
    weights = resolve_base_weights(cases, prompt_names)

    norm_key = int(weights["normalization"])
    if norm_key not in matrices.keyword:
        raise ValueError(
            f"normalization={norm_key} was not cached; add it "
            "to NORMALIZATION_CANDIDATES"
        )

    kwargs = dict(
        intent_weight=weights["intent_weight"],
        bonus=weights["intent_bonus"],
        penalty=weights["intent_penalty"],
        kw_weight=weights["kw_weight"],
        dm_weights=list(weights["dm_weights"]),
        normalization=norm_key,
        centroid_facet_weights=list(weights["centroid_facet_weights"]),
        centroid_bonus=weights["centroid_bonus"],
        centroid_penalty=weights["centroid_penalty"],
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
