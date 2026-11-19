"""Konfiguration, welche Parametergruppen das Tuning optimiert.

Gelesen wird ``tuning_config.yaml`` im code-Root (daneben ``.env`` und
``content_config.yaml``):

.. code-block:: yaml

    tune:
      intent: true          # INTENT_WEIGHT/-BONUS/-PENALTY
      keyword: true         # KEYWORD_WEIGHT + KEYWORD_MATCH_NORMALIZATION
      decision_model: true  # ein DM-Gewicht pro decision_model_prompts-Eintrag
      centroid: true        # CENTROID_FACET_WEIGHTS + MATCH_BONUS/-PENALTY

``false`` hält die Gruppe auf dem Gewichtssnapshot des Caches fest (bei
frisch gebautem Cache: auf den aktuellen .env-Werten). Festgehaltene GROUPEN
erscheinen trotzdem vollständig in der Empfehlung, werden also beim
Übertragen in die .env mitkopiert (in der Regel mit den Werten, die auch
schon drinstehen).

Fehlt die Datei, gilt: alle Gruppen werden getunt. Die Gate-Thresholds sind
keine Gruppe — sie werden aus der Threshold-Landschaft des besten Trials
abgeleitet und stehen immer in der Empfehlung.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TuningConfig:
    """Aktivierte Parametergruppen des Suchlaufs."""

    intent: bool = True
    keyword: bool = True
    decision_model: bool = True
    centroid: bool = True


def load_tuning_config(path: Path) -> TuningConfig:
    """Load tuning_config.yaml; missing file -> all groups on, unknown keys -> error."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return TuningConfig()
    except (OSError, yaml.YAMLError) as ex:
        raise ValueError(f"tuning config {path} is unreadable: {ex}") from ex

    tune = (data or {}).get("tune", {})
    if not isinstance(tune, dict):
        raise ValueError(f"'tune' section in {path} must be a mapping.")

    known = {field.name for field in fields(TuningConfig)}
    unknown = set(tune) - known
    if unknown:
        raise ValueError(
            f"Unknown tuning groups {sorted(unknown)} in {path}; "
            f"valid groups: {sorted(known)}."
        )
    return TuningConfig(**{name: bool(value) for name, value in tune.items()})
