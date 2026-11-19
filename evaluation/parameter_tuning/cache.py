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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config.content_settings import content_config, load_content_config
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
    # Facet name -> (K,) match state per candidate: +1 label match, -1 label
    # mismatch, 0 neutral. Weight-independent, but depends on the cascade
    # floor/margin (snapshotted and validated on load) and the centroids.
    centroid: dict[str, np.ndarray] = field(default_factory=dict)

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

        centroid = await _record_centroid_states(rag_service, question, cands)

        base, bonus, penalty, keyword, dm = [], [], [], {}, []
        for cand in cands:
            breakdown = rr.feature_scores.get(cand.id)
            if breakdown is None:
                raise ValueError(
                    f"feature_scores has no entry for candidate id {cand.id!r} "
                    f"(question: {question!r})"
                )

            # Weight-0-Features sind aus dem Reranker gedroppt (kein
            # Breakdown-Eintrag): nur aktive Features werden gefordert,
            # gedroppte laufen als neutrale Defaults im Cache weiter.
            if settings.INTENT_WEIGHT != 0 and "IntentFeature" not in breakdown:
                raise ValueError(
                    f"feature_scores[{cand.id!r}] lacks 'IntentFeature'; "
                    "reranker output format changed?"
                )
            factor = float(breakdown.get("IntentFeature", 0.0))
            bonus.append(factor > 0)
            penalty.append(factor < 0)

            # Gewichte-0-Prompts sind aus dem Reranker gedroppt und fehlen im
            # Breakdown: nur aktive Prompts werden gefordert, inaktive als 0
            # gecacht (ihr dm-Gewicht pinnt sie ohnehin auf 0).

            missing = [
                p for p, w in zip(prompts, settings.DECISION_MODEL_WEIGHTS)
                if w != 0 and p not in breakdown
            ]
            if missing:
                raise ValueError(
                    f"feature_scores[{cand.id!r}] misses active decision prompts "
                    f"{missing}; module-level content_config may differ from "
                    "CONTENT_CONFIG_FILE"
                )
            dm.append([float(breakdown.get(p, 0.0)) for p in prompts])

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
            centroid=centroid,
        ))

    print()  # close the \r progress line
    return cases


async def _record_centroid_states(rag_service, question: str, cands) -> dict[str, np.ndarray]:
    """Match states (+1/-1/0) per declared facet and candidate.

    The classifier is primed explicitly with a fresh query embedding, so the
    states are independent of the current CENTROID_FACET_WEIGHTS (weight-0
    features are dropped by the reranker and would never prime). The states
    only reflect the cascade's floor/margin — snapshotted and validated on
    load. Requires CENTROID_MATCH_BONUS/_PENALTY > 0, since the sign of the
    raw score distinguishes match from mismatch.
    """
    from app.dependencies.centroids import get_centroid_features

    settings = get_settings()
    features = get_centroid_features()
    if not features:
        return {}
    if settings.CENTROID_MATCH_BONUS <= 0 or settings.CENTROID_MATCH_PENALTY <= 0:
        raise ValueError(
            "CENTROID_MATCH_BONUS and CENTROID_MATCH_PENALTY must be > 0 "
            "while building the tuning cache — the match state is derived "
            "from the sign of the raw feature score."
        )

    embedding = await rag_service.embedding_provider.embed(question)
    for feature in features:
        # Alle Facetten-Features teilen sich einen Classifier; ein Prime
        # klassifiziert jede deklarierte Facette (idempotent pro Query).
        await feature.prime(question, embedding)

    states: dict[str, np.ndarray] = {}
    for feature in features:
        states[feature.facet_name] = np.array(
            [
                # sign of the raw score: +bonus => match, -penalty => mismatch
                1.0 if (raw := feature.score(query=question, candidate=cand)) > 0
                else (-1.0 if raw < 0 else 0.0)
                for cand in cands
            ],
            dtype=float,
        )
    return states


# ------------------------------------------------------------ Persistenz

def save_cache(cases: list[CaseCache], path: Path, sampler_seed: int | None = None) -> None:
    """Persist the case cache so later tuning runs score identical numbers."""
    settings = get_settings()
    payload = {
        "schema": CACHE_SCHEMA,
        # Sampler-Seed dieses Laufes: bei geladenem Cache wiederverwendet,
        # damit dieselbe Matrix + derselbe Seed reproduzierbar bleibt.
        "sampler_seed": sampler_seed,
        "weights_snapshot": weights_snapshot(settings),
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
                "centroid": {
                    name: values.tolist() for name, values in case.centroid.items()
                },
            }
            for case in cases
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def weights_snapshot(settings) -> dict:
    """All weight-like settings that define the cached feature states."""
    return {
        "intent_weight": settings.INTENT_WEIGHT,
        "intent_bonus": settings.INTENT_MATCH_BONUS,
        "intent_penalty": settings.INTENT_MISMATCH_PENALTY,
        "kw_weight": settings.KEYWORD_WEIGHT,
        "normalization": int(settings.KEYWORD_MATCH_NORMALIZATION),
        "dm_weights": list(settings.DECISION_MODEL_WEIGHTS),
        "centroid_facet_weights": list(settings.CENTROID_FACET_WEIGHTS),
        "centroid_bonus": settings.CENTROID_MATCH_BONUS,
        "centroid_penalty": settings.CENTROID_MATCH_PENALTY,
        "centroid_floor": settings.CENTROID_CONFIDENCE_FLOOR,
        "centroid_margin": settings.CENTROID_CONFIDENCE_MARGIN,
        # Der DM-Fallback liefert die Labels für unsichere Facetten und
        # fließt damit in die gecachten Match-Zustände ein.
        "centroid_dm_fallback": bool(settings.CENTROID_DM_FALLBACK),
        # Die DM-Labels hängen an den Facetten-Prompts — Prompt-Wechsel
        # müssen den Cache ungültig machen.
        "centroid_dm_prompts": [
            list(pair) for pair in sorted(
                (name, facet.dm_prompt) for name, facet in content_config.centroid_facets.items()
            )
        ],
    }


def load_cache(
    path: Path,
    rows: list[dict],
    prompts: list[str],
) -> tuple[list[CaseCache] | None, int | None]:
    """Load a previously saved cache with its sampler seed.

    Returns ``(None, None)`` when the cache is missing, stale, or mismatched
    — the caller then rebuilds. Validity checks: same golden rows (order and
    text), all normalization candidates present, the cached decision
    matrix must match the current prompt set, and the cascade floor/margin
    must be unchanged (they are baked into the cached match states). Any
    mismatch forces a fresh build — never tune on silently stale inputs.
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
    current = weights_snapshot(settings)
    if snapshot == current:
        pass
    else:
        stale_keys = [key for key in ("centroid_floor", "centroid_margin",
                                      "centroid_dm_fallback",
                                      "centroid_dm_prompts")
                      if snapshot.get(key) != current[key]]
        if stale_keys:
            # Die gepixelten Match-States hängen an Floor/Margin, dm_fallback
            # und Prompts — andere Werte machen den Cache inhaltlich falsch,
            # keine Randnotiz, sondern rebuild.
            print(
                "  cache was built with stale classification settings "
                f"(changed: {', '.join(stale_keys)}); the match states no "
                "longer match the .env — rebuilding."
            )
            return None, None
        print(
            "  NOTE: cache was built with different .env weights than the "
            "current settings; calibration compares against the snapshot."
        )

    raw_cases = data.get("cases", [])
    if len(raw_cases) != len(rows):
        return None, None

    if "centroid" not in raw_cases[0] or "centroid_facet_weights" not in snapshot:
        print(
            "  cache predates the centroid feature states; rebuilding. "
            "(The new cache also records per-facet match states.)"
        )
        return None, None

    for raw, row in zip(raw_cases, rows):
        if raw.get("question") != row["frage"]:
            return None, None

    cases: list[CaseCache] = []
    for raw in raw_cases:
        keyword = {int(n): np.array(v, dtype=float) for n, v in raw["keyword"].items()}
        missing_norm = [n for n in NORMALIZATION_CANDIDATES if n not in keyword]
        if missing_norm:
            print(f"  cache misses normalization values {missing_norm}; rebuilding.")
            return None, None

        dm = np.array(raw["dm"], dtype=float)
        if dm.shape[1] != len(prompts):
            print(f"  cache was built for a different prompt set; rebuilding.")
            return None, None

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
            # Ältere Caches (pre-Centroid) haben kein "centroid" → leer,
            # was in matrices/scores neutral übersetzt wird. Sobald centroid
            # getunt werden soll, erzwingt dasimplizit einen frischen Build.
            centroid={
                name: np.array(values, dtype=float)
                for name, values in (raw.get("centroid") or {}).items()
            },
        ))
    return cases, data.get("sampler_seed")