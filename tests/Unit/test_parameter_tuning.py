"""Tuning-Gruppen-Konfiguration, Centroid-Matrizen und Cache-Roundtrip."""

import numpy as np
import pytest

from app.config.settings import get_settings
from evaluation.parameter_tuning.cache import (
    CaseCache,
    load_cache,
    save_cache,
    weights_snapshot,
)
from evaluation.parameter_tuning.matrices import (
    FeatureMatrices,
    build_matrices,
    rank_cases,
)
from evaluation.parameter_tuning.search import complete_params
from evaluation.parameter_tuning.tuning_config import (
    TuningConfig,
    load_tuning_config,
)


# ------------------------------------------------------ Konfigurationshelfer

def make_case(question="frage a", expected="1", centroid=None) -> CaseCache:
    return CaseCache(
        question=question,
        category="positiv",
        expected=expected,
        ids=["1#h1", "2#h2"],
        contents=["frage a", "frage b"],
        metadata=[{"question_type": "Wie"}, {"question_type": "Was"}],
        base=np.array([0.5, 0.4]),
        intent_bonus=np.array([True, False]),
        intent_penalty=np.array([False, True]),
        keyword={
            n: np.array([0.1 * n / 2, 0.2 * n / 2])
            for n in (2, 4, 6, 8, 10)
        },
        dm=np.array([[0.6, 0.3], [0.5, 0.4]]),
        reranked_scores=np.array([0.8, 0.7]),
        reranked_top_id="1",
        weights_snapshot=None,
        centroid=centroid or {},
    )


def rows_for(cases: list[CaseCache]) -> list[dict]:
    return [{"frage": case.question} for case in cases]


# ------------------------------------------------------------ TuningConfig

def test_missing_config_enables_all_groups(tmp_path) -> None:
    assert load_tuning_config(tmp_path / "fehlt.yaml") == TuningConfig()


def test_config_disables_groups(tmp_path) -> None:
    path = tmp_path / "tuning_config.yaml"
    path.write_text(
        "tune:\n  intent: false\n  decision_model: false\n",
        encoding="utf-8",
    )
    config = load_tuning_config(path)
    assert config == TuningConfig(
        intent=False, keyword=True, decision_model=False, centroid=True,
    )


def test_config_rejects_unknown_groups(tmp_path) -> None:
    path = tmp_path / "tuning_config.yaml"
    path.write_text("tune:\n  intent: true\n  typografie: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="typografie"):
        load_tuning_config(path)


# ------------------------------------------------------------ complete_params

BASE = {
    "intent_weight": 0.156,
    "intent_bonus": 1.73,
    "intent_penalty": 0.099,
    "normalization": 6,
    "kw_weight": 0.57,
    "dm_weights": [0.0, 0.0, 0.0, 0.0],
    "centroid_facet_weights": [1.0, 1.0],
    "centroid_bonus": 0.15,
    "centroid_penalty": 0.10,
}


def test_complete_params_fills_disabled_groups() -> None:
    # best_params enthält nur gesampelte Werte (DM-Gruppe deaktiviert).
    best_params = {
        "intent_weight": 1.0,
        "intent_bonus": 0.5,
        "intent_penalty": 0.2,
        "keyword_normalization": 4,
        "keyword_weight": 0.3,
        "centroid_bonus": 0.6,
        "centroid_penalty": 0.7,
        "centroid_w_0": 0.9,
        "centroid_w_1": 1.1,
        # Facetten-Zahl im Cache kann von dm_labels abweichen — hier 1 Prompt
    }
    config = TuningConfig(intent=True, keyword=True,
                          decision_model=False, centroid=True)
    p = complete_params(best_params, prompt_names=["p0"], facet_names=["f0", "f1"],
                        config=config, base=BASE)

    # Gesampelte Werte bleiben unangetastet …
    assert p["intent_weight"] == 1.0
    assert p["centroid_w_0"] == 0.9
    # … deaktivierte Gruppen werden auf die Basiswerte gepinnt.
    assert p["dm_0"] == BASE["dm_weights"][0]


def test_complete_params_keeps_sampled_dm_when_enabled() -> None:
    best_params = {"dm_0": 0.42, "dm_1": 0.13}
    config = TuningConfig(intent=False, keyword=False,
                          decision_model=True, centroid=False)
    p = complete_params(best_params, prompt_names=["p0", "p1"], facet_names=["f0"],
                        config=config, base=BASE)
    assert p["dm_0"] == 0.42 and p["dm_1"] == 0.13
    # Pinned groups land on the base snapshot.
    assert p["intent_weight"] == BASE["intent_weight"]
    assert p["keyword_weight"] == BASE["kw_weight"]
    assert p["keyword_normalization"] == BASE["normalization"]
    assert p["centroid_w_0"] == BASE["centroid_facet_weights"][0]


# ------------------------------------------------------- Centroid-Matrizen

def zero_matrices(centroid: dict[str, np.ndarray] | None = None) -> FeatureMatrices:
    return FeatureMatrices(
        base=np.zeros((1, 2)),
        intent_bonus=np.array([[1.0, 0.0]]),
        intent_penalty=np.array([[0.0, 1.0]]),
        keyword={2: np.array([[0.1, 0.2]])},
        dm=np.array([[[0.3], [0.4]]]),
        centroid=centroid or {},
    )


def test_scores_without_centroid_matches_original_semantics() -> None:
    m = zero_matrices()
    scores = m.scores(intent_weight=1.0, bonus=0.2, penalty=0.3, kw_weight=1.0,
                      dm_weights=[2.0], normalization=2)
    expected = (
        0.2 * m.intent_bonus - 0.3 * m.intent_penalty
        + m.keyword[2] + m.dm @ np.array([2.0])
    )
    assert scores == pytest.approx(expected)


def test_centroid_contribution_matches_feature_semantics() -> None:
    m = FeatureMatrices(
        base=np.zeros((1, 3)),
        intent_bonus=np.zeros((1, 3)),
        intent_penalty=np.zeros((1, 3)),
        keyword={2: np.zeros((1, 3))},
        dm=np.zeros((1, 3, 1)),
        centroid={
            "fragetyp": np.array([[1.0, -1.0, 0.0]]),   # match, mismatch, neutral
            "thema": np.array([[-1.0, 1.0, 1.0]]),
        },
    )
    scores = m.scores(
        intent_weight=0.0, bonus=0.0, penalty=0.0, kw_weight=0.0,
        dm_weights=[0.0], normalization=2,
        centroid_facet_weights=[2.0, 1.0],
        centroid_bonus=0.5, centroid_penalty=0.3,
    )
    # Kand. 0: fragetyp +1 → 2*0.5; thema -1 → 1*(-0.3) → 0.7
    # Kand. 1: fragetyp -1 → -0.6; thema +1 → 0.5 → -0.1
    # Kand. 2: fragetyp neutral; thema +1 → 0.5
    assert scores.tolist() == [[pytest.approx(0.7), pytest.approx(-0.1), pytest.approx(0.5)]]


def test_centroid_weight_count_must_match_facets() -> None:
    m = zero_matrices(centroid={"fragetyp": np.array([[1.0, 0.0]])})
    with pytest.raises(ValueError, match="one weight per cached facet"):
        m.scores(intent_weight=0, bonus=0, penalty=0, kw_weight=0, dm_weights=[0.0],
                 normalization=2, centroid_facet_weights=[1.0, 1.0])


def test_build_matrices_pads_facets_from_first_case() -> None:
    case_long = make_case(centroid={"fragetyp": np.array([1.0, -1.0])})
    case_short = make_case(question="frage b", expected="",
                           centroid={"fragetyp": np.array([1.0])})
    m = build_matrices([case_long, case_short])

    assert m.facet_names == ["fragetyp"]
    # Pad-Wert ist neutral (0.0).
    assert m.centroid["fragetyp"][1].tolist() == [1.0, 0.0]


def test_case_without_centroid_states_yields_empty_facets() -> None:
    m = build_matrices([make_case(), make_case(question="frage b", expected="")])
    assert m.facet_names == []
    scores = m.scores(intent_weight=1.0, bonus=0.1, penalty=0.1, kw_weight=0.5,
                      dm_weights=[0.0, 0.0], normalization=2,
                      centroid_facet_weights=None)
    assert scores.shape == (2, 2)


# ------------------------------------------------------- Cache-Roundtrip

SNAPSHOT_SETTINGS = {
    "INTENT_WEIGHT": 0.1,
    "INTENT_MATCH_BONUS": 0.2,
    "INTENT_MISMATCH_PENALTY": 0.3,
    "KEYWORD_WEIGHT": 0.4,
    "KEYWORD_MATCH_NORMALIZATION": 6,
    "DECISION_MODEL_WEIGHTS": [0.5, 0.6],
    "CENTROID_FACET_WEIGHTS": [0.7, 0.8],
    "CENTROID_MATCH_BONUS": 0.15,
    "CENTROID_MATCH_PENALTY": 0.10,
    "CENTROID_CONFIDENCE_FLOOR": 0.30,
    "CENTROID_CONFIDENCE_MARGIN": 0.05,
}


@pytest.fixture
def snapshot_settings(monkeypatch: pytest.MonkeyPatch):
    settings = get_settings()
    for name, value in SNAPSHOT_SETTINGS.items():
        monkeypatch.setattr(settings, name, value)
    return settings


@pytest.mark.asyncio
async def test_cache_round_trip_preserves_centroid_states(
    tmp_path, snapshot_settings,
) -> None:
    cases = [
        make_case(centroid={"fragetyp": np.array([1.0, -1.0]),
                            "thema": np.array([0.0, 1.0])}),
        make_case(question="frage b", expected="",
                  centroid={"fragetyp": np.array([0.0, 0.0]),
                            "thema": np.array([-1.0, 0.0])}),
    ]
    path = tmp_path / "tuning_case_cache.json"
    save_cache(cases, path, sampler_seed=42)

    loaded, seed = load_cache(
        path, rows_for(cases), prompts=list("xy"),
    )
    assert seed == 42
    assert loaded is not None
    for original, restored in zip(cases, loaded):
        np.testing.assert_array_equal(restored.base, original.base)
        np.testing.assert_array_equal(restored.dm, original.dm)
        np.testing.assert_array_equal(
            restored.reranked_scores, original.reranked_scores)
        assert restored.centroid.keys() == original.centroid.keys()
        for facet in original.centroid:
            np.testing.assert_array_equal(
                restored.centroid[facet], original.centroid[facet])
    assert loaded[0].weights_snapshot == weights_snapshot(snapshot_settings)


def test_cache_rejects_changed_cascade_floor(
    tmp_path, snapshot_settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [make_case(centroid={"fragetyp": np.array([1.0, -1.0])})]
    path = tmp_path / "tuning_case_cache.json"
    save_cache(cases, path, sampler_seed=1)

    settings = get_settings()
    monkeypatch.setattr(
        settings, "CENTROID_CONFIDENCE_FLOOR",
        snapshot_settings.CENTROID_CONFIDENCE_FLOOR + 0.1,
    )
    loaded, _ = load_cache(path, rows_for(cases), prompts=list("xy"))
    assert loaded is None


def test_cache_rejects_pre_centroid_caches(tmp_path, snapshot_settings) -> None:
    cases = [make_case()]
    path = tmp_path / "tuning_case_cache.json"
    save_cache(cases, path, sampler_seed=1)
    # Simuliere einen älteren Cache ohne Centroid-Zustände.
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    for raw in data["cases"]:
        raw.pop("centroid")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loaded, _ = load_cache(path, rows_for(cases), prompts=list("xy"))
    assert loaded is None
