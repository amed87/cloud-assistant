"""Phase 2: Optuna search over reranker weights (gate thresholds excluded).

Ranges are deliberately generous — finalize() reports when the optimum sits
at a boundary, which is the signal that a range needs extending.
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler

from .cache import CaseCache, NORMALIZATION_CANDIDATES
from .landscape import gate_landscape
from .matrices import FeatureMatrices, rank_cases

FLOAT_RANGES = {
    "intent_weight": (0.0, 5.0),
    "intent_bonus": (0.0, 2.0),
    "intent_penalty": (0.0, 2.0),   # positive magnitude, subtracted in scores()
    "keyword_weight": (0.0, 1.0),
}
DM_RANGE = (0.0, 1.0)   # per decision-model prompt


def sample_params(trial: optuna.Trial, prompt_names: list[str]) -> dict:
    """Sample one reranker configuration; gate thresholds are NOT sampled."""
    params = {name: trial.suggest_float(name, lo, hi)
              for name, (lo, hi) in FLOAT_RANGES.items()}
    params["keyword_normalization"] = trial.suggest_categorical(
        "keyword_normalization", NORMALIZATION_CANDIDATES)
    for i in range(len(prompt_names)):
        params[f"dm_{i}"] = trial.suggest_float(f"dm_{i}", *DM_RANGE)
    return params


def make_objective(cases: list[CaseCache], matrices: FeatureMatrices,
                   prompt_names: list[str]):
    """Build the Optuna objective: best analytic F0.5 over all gate thresholds."""
    answerable = np.array([bool(c.expected) for c in cases])
    expected = [c.expected for c in cases]

    def objective(trial: optuna.Trial) -> float:
        p = sample_params(trial, prompt_names)
        scores = matrices.scores(
            intent_weight=p["intent_weight"],
            bonus=p["intent_bonus"],
            penalty=p["intent_penalty"],
            kw_weight=p["keyword_weight"],
            dm_weights=[p[f"dm_{i}"] for i in range(len(prompt_names))],
            normalization=p["keyword_normalization"],
        )
        top, gap, predicted = rank_cases(cases, scores)
        correct = np.array([pred == exp if exp else False
                            for pred, exp in zip(predicted, expected)])

        land = gate_landscape(top, gap, correct, answerable)
        trial.set_user_attr("min_score", land.min_score)
        trial.set_user_attr("min_gap", land.min_gap)
        return land.best_f05

    return objective


def run_search(cases, matrices, prompt_names, n_trials: int,
               storage: str | None, study_name: str, seed: int | None) -> optuna.Study:
    """Create/continue a TPE study and optimize the objective."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(make_objective(cases, matrices, prompt_names),
                   n_trials=n_trials, show_progress_bar=True)
    return study