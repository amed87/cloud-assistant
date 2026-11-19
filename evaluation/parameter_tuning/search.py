"""Phase 2: Optuna search over reranker weights (gate thresholds excluded).

Ranges are deliberately generous — finalize() reports when the optimum sits
at a boundary, which is the signal that a range needs extending.

Deaktivierte Gruppen (tuning_config.yaml) halten ihre Parameter auf den
Snapshot-Werten (resolve_base_weights) fest — sie erscheinen trotzdem
unverändert in der Empfehlung.
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler

from .calibration import resolve_base_weights
from .cache import CaseCache, NORMALIZATION_CANDIDATES
from .landscape import gate_landscape
from .matrices import FeatureMatrices, rank_cases
from .tuning_config import TuningConfig

FLOAT_RANGES = {
    "intent_weight": (0.0, 5.0),
    "intent_bonus": (0.0, 2.0),
    "intent_penalty": (0.0, 2.0),   # positive magnitude, subtracted in scores()
    "keyword_weight": (0.0, 1.0),
    "centroid_bonus": (0.0, 1.0),
    "centroid_penalty": (0.0, 1.0),
}
DM_RANGE = (0.0, 1.0)           # per decision-model prompt
CENTROID_WEIGHT_RANGE = (0.0, 2.0)  # per centroid facet


def _pin_disabled(params: dict, prompt_names: list[str],
                  facet_names: list[str], config: TuningConfig, base: dict) -> None:
    """Overwrite disabled-group params with their snapshot values (in place)."""
    if not config.intent:
        for name in ("intent_weight", "intent_bonus", "intent_penalty"):
            params[name] = base[name]
    if not config.keyword:
        params["keyword_normalization"] = base["normalization"]
        params["keyword_weight"] = base["kw_weight"]
    if not config.decision_model:
        for i in range(len(prompt_names)):
            params[f"dm_{i}"] = base["dm_weights"][i]
    if not config.centroid:
        params["centroid_bonus"] = base["centroid_bonus"]
        params["centroid_penalty"] = base["centroid_penalty"]
        for i in range(len(facet_names)):
            params[f"centroid_w_{i}"] = base["centroid_facet_weights"][i]


def sample_params(
    trial: optuna.Trial,
    prompt_names: list[str],
    facet_names: list[str],
    config: TuningConfig,
    base: dict,
) -> dict:
    """Sample enabled groups, take snapshot values for disabled ones."""
    params: dict = {}
    for name in ("intent_weight", "intent_bonus", "intent_penalty"):
        if config.intent:
            params[name] = trial.suggest_float(name, *FLOAT_RANGES[name])
    params["keyword_normalization"] = trial.suggest_categorical(
        "keyword_normalization", NORMALIZATION_CANDIDATES) if config.keyword else None
    if config.keyword:
        params["keyword_weight"] = trial.suggest_float("keyword_weight", *FLOAT_RANGES["keyword_weight"])
    if config.decision_model:
        for i in range(len(prompt_names)):
            params[f"dm_{i}"] = trial.suggest_float(f"dm_{i}", *DM_RANGE)
    if config.centroid:
        params["centroid_bonus"] = trial.suggest_float("centroid_bonus", *FLOAT_RANGES["centroid_bonus"])
        params["centroid_penalty"] = trial.suggest_float("centroid_penalty", *FLOAT_RANGES["centroid_penalty"])
        for i in range(len(facet_names)):
            params[f"centroid_w_{i}"] = trial.suggest_float(
                f"centroid_w_{i}", *CENTROID_WEIGHT_RANGE)
    _pin_disabled(params, prompt_names, facet_names, config, base)
    return params


def complete_params(
    best_params: dict,
    prompt_names: list[str],
    facet_names: list[str],
    config: TuningConfig,
    base: dict,
) -> dict:
    """Restore disabled-group values on an Optuna best_params dict.

    Optuna stores only sampled params; disabled groups never enter
    best_params, but scores_kwargs() needs a complete dict.
    """
    p = dict(best_params)
    _pin_disabled(p, prompt_names, facet_names, config, base)
    return p


def scores_kwargs(params: dict, prompt_names: list[str], facet_names: list[str]) -> dict:
    """Translate a search parameter dict into FeatureMatrices.scores kwargs."""
    return dict(
        intent_weight=params["intent_weight"],
        bonus=params["intent_bonus"],
        penalty=params["intent_penalty"],
        kw_weight=params["keyword_weight"],
        dm_weights=[params[f"dm_{i}"] for i in range(len(prompt_names))],
        normalization=params["keyword_normalization"],
        centroid_facet_weights=[params[f"centroid_w_{i}"] for i in range(len(facet_names))],
        centroid_bonus=params["centroid_bonus"],
        centroid_penalty=params["centroid_penalty"],
    )


def make_objective(cases: list[CaseCache], matrices: FeatureMatrices,
                   prompt_names: list[str], config: TuningConfig, base: dict):
    """Build the Optuna objective: best analytic F0.5 over all gate thresholds."""
    answerable = np.array([bool(c.expected) for c in cases])
    expected = [c.expected for c in cases]
    facet_names = matrices.facet_names

    def objective(trial: optuna.Trial) -> float:
        p = sample_params(trial, prompt_names, facet_names, config, base)
        scores = matrices.scores(**scores_kwargs(p, prompt_names, facet_names))
        top, gap, predicted = rank_cases(cases, scores)
        correct = np.array([pred == exp if exp else False
                            for pred, exp in zip(predicted, expected)])

        land = gate_landscape(top, gap, correct, answerable)
        trial.set_user_attr("min_score", land.min_score)
        trial.set_user_attr("min_gap", land.min_gap)
        return land.best_f05

    return objective


def run_search(cases, matrices, prompt_names, n_trials: int,
               storage: str | None, study_name: str, seed: int | None,
               config: TuningConfig, base: dict) -> optuna.Study:
    """Create/continue a TPE study and optimize the objective."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(
        make_objective(cases, matrices, prompt_names, config, base),
        n_trials=n_trials, show_progress_bar=True)
    return study
