"""Phase 0: run retrieval once per golden case and cache all feature inputs.

Raw feature scores are taken directly from the live reranker output
(``RetrievalResult.feature_scores``) instead of being recomputed, so the
tuner scores the very same numbers production produced. Missing values
abort loudly - silent 0.0 defaults are forbidden here on purpose.

Der Cache wird als JSON-Datei persistiert: wiederholte Tuning-Läufe nutzen
dann exakt dieselben DM-/Keyword-/Intent-Werte statt bei jedem Lauf frische
Ollama-Antworten zu würfeln (DM-Jitter war die Quelle unstabiler Optima).
Der Cache ist proaktiv zu löschen bzw. mit ``--refresh-cache`` neu zu bauen,
sobald sich FAQ-Datenbank, Features oder Code ändern.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config.content_settings import load_content_config
from app.config.settings import get_settings
from app.quality.ranking.keyword_feature import KeywordFeature

NORMALIZATION_CANDIDATES = [2, 4, 6, 8, 10]

CACHE_SCHEMA = 1

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
    weights_snapshot: dict | None = None  # settings weights at cache build

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


# ------------------------------------------------------------ Persistenz

def save_cache(cases: list[CaseCache], path: Path, sampler_seed: int | None = None) -> None:
    """Persist the case cache so later tuning runs score identical numbers."""
    settings = get_settings()
    payload = {
        "schema": CACHE_SCHEMA,
        # Sampler-Seed dieses Laufes: bei geladenem Cache wiederverwendet,
        # damit dieselbe Matrix + derselbe Seed reproduzierbar bleibt.
        "sampler_seed": sampler_seed,
        "weights_snapshot": {
            "intent_weight": settings.INTENT_WEIGHT,
            "intent_bonus": settings.INTENT_MATCH_BONUS,
            "intent_penalty": settings.INTENT_MISMATCH_PENALTY,
            "kw_weight": settings.KEYWORD_WEIGHT,
            "normalization": int(settings.KEYWORD_MATCH_NORMALIZATION),
            "dm_weights": list(settings.DECISION_MODEL_WEIGHTS),
        },
        "n_cases": len(cases),
        "cases": [
            {
                "question": case.question,
                "category": case.category,
                "expected": case.expected,
                "ids": case.ids,
                "contents": case.contents,
                "metadata": case.metadata,
                "base": case.base.tolist(),
                "intent_bonus": case.intent_bonus.tolist(),
                "intent_penalty": case.intent_penalty.tolist(),
                "keyword": {str(n): values.tolist() for n, values in case.keyword.items()},
                "dm": case.dm.tolist(),
                "reranked_scores": case.reranked_scores.tolist(),
                "reranked_top_id": case.reranked_top_id,
            }
            for case in cases
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_cache(
    path: Path,
    rows: list[dict],
    prompts: list[str],
) -> tuple[list[CaseCache] | None, int | None]:
    """Load a previously saved cache with its sampler seed.

    Returns ``(None, None)`` when the cache is missing, stale, or mismatched
    — the caller then rebuilds. Validity checks: same golden rows (order and
    text), all normalization candidates present, and the cached decision
    matrix must match the current prompt set. Any mismatch forces a fresh
    build — never tune on silently stale inputs.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, json.JSONDecodeError) as ex:
        print(f"  cache file {path} unreadable ({ex}); rebuilding.")
        return None, None

    if data.get("schema") != CACHE_SCHEMA:
        print(f"  cache file {path} has an outdated schema; rebuilding.")
        return None, None

    snapshot = data.get("weights_snapshot")
    if not snapshot:
        print("  cache has no weights snapshot; rebuilding (delete file or pass --refresh-cache).")
        return None, None

    settings = get_settings()
    current = {
        "intent_weight": settings.INTENT_WEIGHT,
        "intent_bonus": settings.INTENT_MATCH_BONUS,
        "intent_penalty": settings.INTENT_MISMATCH_PENALTY,
        "kw_weight": settings.KEYWORD_WEIGHT,
        "normalization": int(settings.KEYWORD_MATCH_NORMALIZATION),
        "dm_weights": list(settings.DECISION_MODEL_WEIGHTS),
    }
    if snapshot != current:
        print(
            "  NOTE: cache was built with different .env weights than the "
            "current settings; calibration compares against the snapshot."
        )

    raw_cases = data.get("cases", [])
    if len(raw_cases) != len(rows):
        return None

    for raw, row in zip(raw_cases, rows):
        if raw.get("question") != row["frage"]:
            return None

    cases: list[CaseCache] = []
    for raw in raw_cases:
        keyword = {int(n): np.array(v, dtype=float) for n, v in raw["keyword"].items()}
        missing_norm = [n for n in NORMALIZATION_CANDIDATES if n not in keyword]
        if missing_norm:
            print(f"  cache misses normalization values {missing_norm}; rebuilding.")
            return None

        dm = np.array(raw["dm"], dtype=float)
        if dm.shape[1] != len(prompts):
            print(f"  cache was built for a different prompt set; rebuilding.")
            return None

        cases.append(CaseCache(
            question=raw["question"],
            category=raw["category"],
            expected=raw["expected"],
            ids=raw["ids"],
            contents=raw["contents"],
            metadata=raw["metadata"],
            base=np.array(raw["base"], dtype=float),
            intent_bonus=np.array(raw["intent_bonus"], dtype=bool),
            intent_penalty=np.array(raw["intent_penalty"], dtype=bool),
            keyword=keyword,
            dm=dm,
            reranked_scores=np.array(raw["reranked_scores"], dtype=float),
            reranked_top_id=raw["reranked_top_id"],
            weights_snapshot=snapshot,
        ))
    return cases, data.get("sampler_seed")