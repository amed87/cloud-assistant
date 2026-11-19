"""Phase 0: run retrieval once per golden case and cache all feature inputs.

Raw feature scores are taken directly from the live reranker output
(``RetrievalResult.feature_scores``) instead of being recomputed, so the
tuner scores the very same numbers production produced. Missing values
abort loudly - silent 0.0 defaults are forbidden here on purpose.
"""

from dataclasses import dataclass

import numpy as np

from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.quality.ranking.keyword_feature import KeywordFeature

NORMALIZATION_CANDIDATES = [2, 4, 6, 8, 10]

@dataclass
class CaseCache:
    """Cached feature inputs for one golden case."""

    question: str
    category: str
    expected: str                  # bare FAQ id, "" if unanswerable
    ids: list[str]                 # full candidate ids (with version suffix)
    contents: list[str]
    metadata: list[dict]
    base: np.ndarray               # (K,) chroma scores
    intent_bonus: np.ndarray       # (K,) bool: candidate got the intent bonus
    intent_penalty: np.ndarray     # (K,) bool: candidate got the intent penalty
    keyword: dict[int, np.ndarray] # normalization value -> (K,) raw factors
    dm: np.ndarray                 # (K, P) raw probabilities per prompt
    reranked_scores: np.ndarray    # (K,) live final scores, aligned to ids
    reranked_top_id: str | None    # bare id of the live top-1 candidate

async def build_cache(rows: list[dict], rag_service) -> list[CaseCache]:
    """Retrieve every golden case once and extract all feature inputs."""
    settings = get_settings()
    content_config = load_content_config(path=settings.CONTENT_CONFIG_FILE)
    prompts = list(content_config.decision_model_prompts)
    cases: list[CaseCache] = []

    print(f"Retrieving {len(rows)} golden cases (expensive, runs once)...")

    for i, row in enumerate(rows, start=1):
        question = row["frage"]
        expected = row["erwartete_entry_id"].strip()

        rr = await rag_service.quick_retrieve(question=question, eval=True)
        cands = rr.chroma

        print(f"\r  case {i}/{len(rows)}", end="", flush=True)

        base, bonus, penalty, keyword, dm = [], [], [], {}, []
        for cand in cands:
            breakdown = rr.feature_scores.get(cand.id)
            if breakdown is None:
                raise ValueError(
                    f"feature_scores has no entry for candidate id {cand.id!r} "
                    f"(question: {question!r})"
                )

            if "IntentFeature" not in breakdown:
                raise ValueError(
                    f"feature_scores[{cand.id!r}] lacks 'IntentFeature'; "
                    "reranker output format changed?"
                )
            factor = float(breakdown["IntentFeature"])
            bonus.append(factor > 0)
            penalty.append(factor < 0)

            missing = [p for p in prompts if p not in breakdown]
            if missing:
                raise ValueError(
                    f"feature_scores[{cand.id!r}] misses decision prompts "
                    f"{missing}; module-level content_config may differ from "
                    "CONTENT_CONFIG_FILE"
                )
            dm.append([float(breakdown[p]) for p in prompts])

            for n in NORMALIZATION_CANDIDATES:
                kf = KeywordFeature(weight=1.0, normalization=n)
                keyword.setdefault(n, []).append(
                    kf.score(query=question, candidate=cand)
                )

            base.append(float(cand.score))

        # Live final scores, aligned to the chroma candidate order.
        live_scores = {r.id: float(r.score) for r in rr.reranker}
        unmatched = [c.id for c in cands if c.id not in live_scores]
        if unmatched:
            raise ValueError(
                f"reranker output missing candidates {unmatched} "
                f"(question: {question!r})"
            )
        reranked = [live_scores[c.id] for c in cands]

        top = rr.reranker[0] if rr.reranker else None
        cases.append(CaseCache(
            question=question,
            category=row["kategorie"],
            expected=expected,
            ids=[c.id for c in cands],
            contents=[c.content for c in cands],
            metadata=[c.metadata for c in cands],
            base=np.array(base, dtype=float),
            intent_bonus=np.array(bonus, dtype=bool),
            intent_penalty=np.array(penalty, dtype=bool),
            keyword={n: np.array(v, dtype=float) for n, v in keyword.items()},
            dm=np.array(dm, dtype=float),
            reranked_scores=np.array(reranked, dtype=float),
            reranked_top_id=top.id.partition(":")[0] if top else None,
        ))

    print()  # close the \r progress line
    return cases