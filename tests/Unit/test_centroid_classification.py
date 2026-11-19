"""Centroid store, cascade classifier, match feature, and feature factory."""

import json

import pytest

from app.config.content_settings import CentroidFacet
from app.documents.models import FAQEntry
from app.quality.classification.centroid_store import CentroidStore
from app.quality.classification.query_classifier import (
    FacetClassifierConfig,
    QueryFacetClassifier,
)
from app.quality.ranking.centroid_feature import (
    CentroidMatchFeature,
    build_centroid_features,
)
from app.vectorstores.models import SearchResult


# ------------------------------------------------------------------ Helfer

class FakeEmbeddingProvider:
    """Deterministic embeddings: label char drives a one-hot-ish vector."""

    def __init__(self) -> None:
        self.model = "fake-embed"
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        return list(vectors[text[-1]])


class FakeDecisionProvider:
    def __init__(self, result: dict[str, float] | None = None, fails: bool = False):
        self.calls: list[tuple[str, dict[str, str], list[str]]] = []
        self.result = result
        self.fails = fails

    async def decide(self, query, questions, choices):
        self.calls.append((query, questions, choices))
        if self.fails:
            raise RuntimeError("provider down")
        return {
            name: type("R", (), {"probabilities": self.result or {}})()
            for name in questions
        }


def entry(entry_id: str, question: str, question_type: str, subjects: list[str]) -> FAQEntry:
    return FAQEntry(
        id=entry_id,
        question=question,
        answer="answer",
        subjects=subjects,
        question_type=question_type,
        keywords=[],
    )


def make_store(tmp_path, provider: FakeEmbeddingProvider) -> CentroidStore:
    facets = {
        "fragetyp": CentroidFacet(metadata_key="question_type", fallback="Generell"),
        "thema": CentroidFacet(metadata_key="subjects", multi_label=True, fallback="Allgemein"),
    }
    return CentroidStore(
        store_path=tmp_path / "centroids.json",
        embedding_provider=provider,
        facets=facets,
    )


def make_facets() -> dict[str, CentroidFacet]:
    return {
        "fragetyp": CentroidFacet(metadata_key="question_type", fallback="Generell"),
    }


# -------------------------------------------------------- CentroidStore

@pytest.mark.asyncio
async def test_store_rebuild_groups_by_label_and_persists(tmp_path) -> None:
    provider = FakeEmbeddingProvider()
    store = make_store(tmp_path, provider)
    entries = [
        entry("id1#hash", "frage a", "Wie", ["ThemaA"]),
        entry("id2#hash", "frage a", "Wie", ["ThemaA", "ThemaB"]),
        entry("id3#hash", "frage b", "Was", ["ThemaB"]),
    ]

    await store.rebuild(entries)

    # Multi-Label: entry 2 contributes to both subjects.
    assert store.labels("fragetyp").keys() == {"Wie", "Was"}
    assert store.labels("thema").keys() == {"ThemaA", "ThemaB"}
    # Mean over [1,0]/[1,0]/[0,1] vectors for ThemaA = [1.0, 0.0]
    assert store.labels("thema")["ThemaA"] == pytest.approx([1.0, 0.0])
    # Mean over [1,0]/[0,1] for ThemaB = [0.5, 0.5]
    assert store.labels("thema")["ThemaB"] == pytest.approx([0.5, 0.5])

    reloaded = make_store(tmp_path, FakeEmbeddingProvider())
    assert reloaded.load() is True
    assert reloaded.labels("fragetyp").keys() == {"Wie", "Was"}


@pytest.mark.asyncio
async def test_store_rebuild_reuses_cached_embeddings(tmp_path) -> None:
    provider = FakeEmbeddingProvider()
    store = make_store(tmp_path, provider)
    entries = [entry("id1#hash", "frage a", "Wie", ["ThemaA"])]

    await store.rebuild(entries)
    first_calls = len(provider.calls)

    await store.rebuild(entries)
    assert len(provider.calls) == first_calls  # nothing re-embedded

    entries_changed = [entry("id2#newhash", "frage a", "Wie", ["ThemaA"])]
    await store.rebuild(entries_changed)
    assert len(provider.calls) == first_calls + 1  # only the new/changed entry
    assert store.labels("fragetyp").keys() == {"Wie"}


def test_store_load_degrades_cleanly(tmp_path) -> None:
    provider = FakeEmbeddingProvider()
    store = make_store(tmp_path, provider)

    # Missing file.
    assert store.load() is False
    assert store.labels("fragetyp") == {}

    # Wrong embedding model.
    payload = {"schema": 1, "embedding_model": "other-model", "facets": {}}
    (tmp_path / "centroids.json").write_text(json.dumps(payload), encoding="utf-8")
    assert store.load() is False

    # Broken JSON.
    (tmp_path / "centroids.json").write_text("{not json", encoding="utf-8")
    assert store.load() is False


@pytest.mark.asyncio
async def test_store_rebuild_keeps_prior_file_on_empty_input(tmp_path) -> None:
    provider = FakeEmbeddingProvider()
    store = make_store(tmp_path, provider)
    await store.rebuild([entry("id1#hash", "frage a", "Wie", ["ThemaA"])])
    file_content = (tmp_path / "centroids.json").read_text(encoding="utf-8")

    await store.rebuild([])

    assert (tmp_path / "centroids.json").read_text(encoding="utf-8") == file_content


# ------------------------------------------------ QueryFacetClassifier

def make_classifier(
    tmp_path,
    provider: FakeEmbeddingProvider | None = None,
    decision_provider=None,
    floor=0.30,
    margin=0.05,
    dm_fallback=False,
    facets: dict[str, CentroidFacet] | None = None,
) -> QueryFacetClassifier:
    if provider is None:
        provider = FakeEmbeddingProvider()
    store = make_store(tmp_path, provider)
    facets = facets or {
        "fragetyp": CentroidFacet(
            metadata_key="question_type",
            fallback="Generell",
            dm_prompt="Typ?",
        ),
    }
    return QueryFacetClassifier(
        store=store,
        facets=facets,
        config=FacetClassifierConfig(
            floor=floor,
            margin=margin,
            dm_fallback=dm_fallback,
        ),
        decision_provider=decision_provider,
    )


@pytest.mark.asyncio
async def test_classifier_confident_via_centroid(tmp_path) -> None:
    classifier = make_classifier(tmp_path, FakeEmbeddingProvider())
    entries = [entry("id1#hash", "frage a", "Wie", ["ThemaA"])]
    await classifier._store.rebuild(entries)

    # Max cosine to the "Wie" centroid (a-ish vector also equals query embedding).
    await classifier.prime("frage a", [1.0, 0.0])

    prediction = classifier.prediction("fragetyp")
    assert prediction is not None
    assert prediction.label == "Wie"
    assert prediction.source == "centroid"


@pytest.mark.asyncio
async def test_classifier_uncertain_without_dm_uses_fallback(tmp_path) -> None:
    classifier = make_classifier(tmp_path, FakeEmbeddingProvider())
    entries = [entry("id1#hash", "frage a", "Wie", ["ThemaA"])]
    await classifier._store.rebuild(entries)

    # Equidistant to both labels → no confident top1.
    await classifier.prime("frage a", [-1.0, 0.0])  # cosine -1 → uncertain

    prediction = classifier.prediction("fragetyp")
    assert prediction is not None
    assert prediction.label == "Generell"
    assert prediction.source == "fallback"


@pytest.mark.asyncio
async def test_classifier_dm_fallback_called_once_per_uncertain_facet(tmp_path) -> None:
    dm = FakeDecisionProvider(result={"Wie": 0.7, "Was": 0.3})
    classifier = make_classifier(
        tmp_path, FakeEmbeddingProvider(), decision_provider=dm, dm_fallback=True,
    )
    entries = [entry("id1#hash", "frage a", "Wie", ["ThemaA"])]
    await classifier._store.rebuild(entries)

    await classifier.prime("frage a", [-1.0, 0.0])  # cosine -1 → uncertain

    assert len(dm.calls) == 1
    assert dm.calls[0][1] == {"fragetyp": "Typ?"}
    assert dm.calls[0][2] == ["Wie"]  # choices are the facet's centroid labels
    assert classifier.prediction("fragetyp").label == "Wie"
    assert classifier.prediction("fragetyp").source == "decision_model"

    # Second prime of the same query is idempotent — no further DM call.
    await classifier.prime("frage a", [-1.0, 0.0])  # cosine -1 → uncertain
    assert len(dm.calls) == 1


@pytest.mark.asyncio
async def test_classifier_dm_failure_degrades_to_fallback(tmp_path) -> None:
    dm = FakeDecisionProvider(fails=True)
    classifier = make_classifier(
        tmp_path, FakeEmbeddingProvider(), decision_provider=dm, dm_fallback=True,
    )
    entries = [entry("id1#hash", "frage a", "Wie", ["ThemaA"])]
    await classifier._store.rebuild(entries)

    await classifier.prime("frage a", [-1.0, 0.0])  # cosine -1 → uncertain

    prediction = classifier.prediction("fragetyp")
    assert prediction.label == "Generell"
    assert prediction.source == "fallback"


@pytest.mark.asyncio
async def test_classifier_without_embedding_leaves_no_prediction(tmp_path) -> None:
    classifier = make_classifier(tmp_path, FakeEmbeddingProvider())

    await classifier.prime("frage a", None)

    # Without an embedding the features vote neutrally (no prediction).
    assert classifier.prediction("fragetyp") is None


# ------------------------------------------------- CentroidMatchFeature

def candidate(metadata: dict) -> SearchResult:
    return SearchResult(id="c1", content="content", score=1.0, metadata=metadata)


@pytest.mark.asyncio
async def test_match_feature_bonus_penalty_and_neutral(tmp_path) -> None:
    both_facets = {
        "fragetyp": CentroidFacet(metadata_key="question_type", fallback="Generell"),
        "thema": CentroidFacet(metadata_key="subjects", multi_label=True, fallback="Allgemein"),
    }
    classifier = make_classifier(tmp_path, FakeEmbeddingProvider(), facets=both_facets)
    entries = [entry("id1#hash", "frage a", "Wie", ["ThemaA"])]
    await classifier._store.rebuild(entries)
    await classifier.prime("frage a", [1.0, 0.0])  # "Wie" + "ThemaA", confident

    feature = CentroidMatchFeature(
        weight=1.0,
        classifier=classifier,
        facet=classifier.facets["fragetyp"],
        facet_name="fragetyp",
        bonus=0.5,
        penalty=0.25,
    )

    assert feature.name == "Centroid:fragetyp"
    # Exact label match → bonus.
    assert feature.score("frage a", candidate({"question_type": "Wie"})) == pytest.approx(0.5)
    # Confident different (non-fallback) label → penalty.
    assert feature.score("frage a", candidate({"question_type": "Was"})) == pytest.approx(-0.25)
    # Fallback label on the candidate side means "unknown" → neutral.
    assert feature.score("frage a", candidate({"question_type": "Generell"})) == 0.0
    # Missing metadata on the candidate → neutral.
    assert feature.score("frage a", candidate({})) == 0.0

    # Multi-label: query label in candidate labels → bonus.
    thema_feature = CentroidMatchFeature(
        weight=1.0,
        classifier=classifier,
        facet=both_facets["thema"],
        facet_name="thema",
        bonus=0.15,
        penalty=0.10,
    )
    assert thema_feature.score("frage a", candidate({"subjects": ["ThemaX", "ThemaA"]})) == 0.15
    assert thema_feature.score("frage a", candidate({"subjects": ["ThemaX"]})) == -0.10
    # Fallback label among the candidate labels → neutral.
    assert thema_feature.score("frage a", candidate({"subjects": ["Allgemein"]})) == 0.0


@pytest.mark.asyncio
async def test_match_feature_neutral_without_prediction(tmp_path) -> None:
    classifier = make_classifier(tmp_path, FakeEmbeddingProvider())
    # No prime() → no prediction available.
    feature = CentroidMatchFeature(
        weight=1.0,
        classifier=classifier,
        facet=classifier.facets["fragetyp"],
        facet_name="fragetyp",
    )
    assert feature.score("frage a", candidate({"question_type": "Wie"})) == 0.0


# ------------------------------------------------- build_centroid_features

def test_build_centroid_features_order_and_mismatch(tmp_path) -> None:
    classifier = make_classifier(tmp_path)

    facets = {
        "fragetyp": CentroidFacet(metadata_key="question_type", fallback="Generell"),
        "thema": CentroidFacet(metadata_key="subjects", multi_label=True),
        "extra": CentroidFacet(metadata_key="keywords"),
    }
    features = build_centroid_features(
        classifier=classifier,
        weights=[1.0, 0.5, 0.0],
        bonus=0.2,
        penalty=0.3,
        facets=facets,
    )

    assert [feature.weight for feature in features] == [1.0, 0.5, 0.0]
    assert [feature.facet_name for feature in features] == ["fragetyp", "thema", "extra"]
    assert features[0].bonus == pytest.approx(0.2)
    assert features[0].penalty == pytest.approx(0.3)

    with pytest.raises(ValueError):
        build_centroid_features(
            classifier=classifier,
            weights=[1.0],
            bonus=0.2,
            penalty=0.3,
            facets=facets,
        )


def test_build_centroid_features_defaults_to_classifier_facets(tmp_path) -> None:
    classifier = make_classifier(tmp_path, FakeEmbeddingProvider())
    features = build_centroid_features(
        classifier=classifier,
        weights=[1.0],
        bonus=0.1,
        penalty=0.1,
    )
    assert [feature.facet_name for feature in features] == ["fragetyp"]


# ------------------------------------------------------- Reranker drop

@pytest.mark.asyncio
async def test_reranker_drops_zero_weight_centroid_feature(tmp_path) -> None:
    classifier = make_classifier(tmp_path, FakeEmbeddingProvider())
    entries = [entry("id1#hash", "frage a", "Wie", ["ThemaA"])]
    await classifier._store.rebuild(entries)

    feature = CentroidMatchFeature(
        weight=0.0,
        classifier=classifier,
        facet=classifier.facets["fragetyp"],
        facet_name="fragetyp",
    )
    from app.quality.ranking.reranker import Reranker

    reranker = Reranker(features=[feature])
    assert feature not in reranker._features

    # Zero-weight feature must not trigger classification (prime skipped).
    await reranker.rerank([candidate({"question_type": "Wie"})], "frage a")
    assert classifier.predictions == {}
