"""Phase 1: stack cached per-case features into dense numpy arrays."""

from dataclasses import dataclass

import numpy as np

from .cache import CaseCache


@dataclass
class FeatureMatrices:
    """Dense feature arrays over all cases (padded to the widest case)."""

    base: np.ndarray               # (C, K)
    intent_bonus: np.ndarray       # (C, K) float
    intent_penalty: np.ndarray     # (C, K) float
    keyword: dict[int, np.ndarray] # normalization -> (C, K)
    dm: np.ndarray                 # (C, K, P)

    def scores(
        self,
        intent_weight: float,
        bonus: float,
        penalty: float,
        kw_weight: float,
        dm_weights: list[float],
        normalization: int,
    ) -> np.ndarray:
        """Weighted-sum scores; bonus adds, positive penalty subtracts."""
        return (
            self.base
            + intent_weight * (bonus * self.intent_bonus - penalty * self.intent_penalty)
            + kw_weight * self.keyword[normalization]
            + self.dm @ np.asarray(dm_weights, dtype=float)
        )


def build_matrices(cases: list[CaseCache]) -> FeatureMatrices:
    """Pad ragged per-case arrays into dense (C, K, ...) matrices."""
    kmax = max(len(c.ids) for c in cases)
    n = len(cases)
    n_prompts = cases[0].dm.shape[1]

    def pad(arr: np.ndarray, fill: float) -> np.ndarray:
        out = np.full((kmax,) + arr.shape[1:], fill, dtype=float)
        out[: len(arr)] = arr
        return out

    base = np.stack([pad(c.base, -np.inf) for c in cases])
    bonus = np.stack([pad(c.intent_bonus.astype(float), 0.0) for c in cases])
    penalty = np.stack([pad(c.intent_penalty.astype(float), 0.0) for c in cases])
    dm = np.stack([pad(c.dm, 0.0) for c in cases])
    keyword = {
        norm: np.stack([pad(c.keyword[norm], 0.0) for c in cases])
        for norm in cases[0].keyword
    }

    m = FeatureMatrices(base=base, intent_bonus=bonus, intent_penalty=penalty,
                        keyword=keyword, dm=dm)
    assert m.dm.shape == (n, kmax, n_prompts)
    return m


def rank_cases(cases: list[CaseCache], scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (top_score, gap, predicted_bare_id) per case from a score matrix."""
    order = np.argsort(-scores, axis=1, kind="stable")
    top = np.take_along_axis(scores, order[:, :1], axis=1)[:, 0]
    second = np.take_along_axis(scores, order[:, 1:2], axis=1)[:, 0]
    predicted = [cases[i].ids[order[i, 0]].partition(":")[0] for i in range(len(cases))]
    return top, top - second, predicted