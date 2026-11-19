"""Persistente Zentroid-Verwaltung für qualitätsseitige Query-Klassifikation.

Pro Facette (z. B. Fragetyp, Thema) wird ein Zentroid pro Label aus den
Frage-Embeddings aller FAQ-Einträge mit diesem Label gebildet. Zur Queryzeit
genügt eine Cosine-Ähnlichkeit der (ohnehin berechneten) Query-Embedding
gegen diese Zentroide — keine LLM-Aufrufe.

Persistenz liegt als ``centroids.json`` unter ``DB_PATH`` (unmittelbar neben
der Chroma-DB), damit Update und Server die Daten über Prozessgrenzen teilen
können. Der Store cached die Frage-Embeddings pro Entry-ID; da die ID einen
Content-Hash trägt, re-embedden Updates nur neue oder geänderte Fragen.

Erweiterbarkeit: Eine neue Facette (z. B. ein weiteres Metadaten-Feld) ist
ein ``centroid_facets``-Eintrag in content_config.yaml — hier läuft der Code
generisch über die deklarierte Liste.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config.content_settings import CentroidFacet
from app.documents.models import FAQEntry
from app.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(slots=True)
class LabelPrediction:
    """Query-seitige Label-Vorhersage für eine Facette."""

    label: str | None
    confidence: float      # Cosine to the label or DM probability
    source: str            # "centroid" | "decision_model" | "fallback"


class CentroidStore:
    """Baut, lädt und persistiert Zentroide je Facette und Label."""

    def __init__(
        self,
        store_path: Path,
        embedding_provider: EmbeddingProvider,
        facets: dict[str, CentroidFacet],
    ) -> None:
        self._path = store_path
        self._provider = embedding_provider
        self._facets = facets
        self._model = getattr(embedding_provider, "model", "unknown")
        # In-Memory-Zustand
        self._entry_embeddings: dict[str, list[float]] = {}
        self._centroids: dict[str, dict[str, list[float]]] = {}
        self._source_hash: str = ""
        self._loaded = False
        self._mtime: int | None = None

    # ------------------------------------------------------------- Laden

    def load(self) -> bool:
        """Load the persisted centroids; invalid files degrade to an empty store."""
        self._loaded = True
        try:
            self._mtime = self._path.stat().st_mtime_ns
        except OSError:
            self._mtime = None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("Centroid file %s not found; features stay neutral until the next update.", self._path)
            return False
        except (OSError, json.JSONDecodeError) as ex:
            logger.error("Failed to read centroid file %s: %s", self._path, ex)
            return False

        if data.get("schema") != SCHEMA_VERSION:
            logger.warning("Centroid file %s has an incompatible schema; rebuilding it requires an update run.", self._path)
            return False
        if data.get("embedding_model") != self._model:
            logger.warning(
                "Centroid file %s was built with model %r, current model %r — ignoring until rebuilt.",
                self._path, data.get("embedding_model"), self._model,
            )
            return False

        self._source_hash = data.get("source_hash", "")
        self._entry_embeddings = data.get("entry_embeddings", {})
        self._centroids = data.get("facets", {})
        return True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            return
        # Externe Aktualisierung (Update-Service, anderer Prozess): Reload,
        # sobald die Datei sich geändert hat — ein stat() pro Query ist billig.
        try:
            mtime = self._path.stat().st_mtime_ns
        except OSError:
            return
        if mtime != self._mtime:
            logger.info("Centroid file changed on disk (%s); reloading.", self._path)
            self.load()

    def reload(self) -> None:
        """Discard the in-memory state and read the file again (after updates)."""
        self._loaded = False
        self._entry_embeddings = {}
        self._centroids = {}
        self._source_hash = ""
        self.load()

    # ------------------------------------------------------------ Zugriff

    def is_available(self, facet: str) -> bool:
        """True when at least one label centroid exists for the facet."""
        self._ensure_loaded()
        return bool(self._centroids.get(facet))

    def labels(self, facet: str) -> dict[str, list[float]]:
        """Label → Zentroid-Vektor einer Facette (leer, wenn nicht verfügbar)."""
        self._ensure_loaded()
        return self._centroids.get(facet, {})

    # ------------------------------------------------------------ Aufbau

    async def rebuild(self, entries: list[FAQEntry]) -> None:
        """Compute and persist all facet centroids from the given entries."""
        entry_embeddings = dict(self._entry_embeddings)
        current_ids: set[str] = set()

        for entry in entries:
            current_ids.add(entry.id)
            if entry.id not in entry_embeddings:
                entry_embeddings[entry.id] = await self._provider.embed(entry.question)

        # Vergessene Embeddings aufräumen — ID enthält den Content-Hash,
        # geänderte Einträge tauchen unter neuer ID auf.
        entry_embeddings = {
            entry_id: vector
            for entry_id, vector in entry_embeddings.items()
            if entry_id in current_ids
        }
        if not entries:
            logger.warning("Centroid rebuild got an empty FAQ document; keeping prior centroids file untouched.")
            return

        centroids: dict[str, dict[str, list[float]]] = {}
        for name, facet in self._facets.items():
            grouped: dict[str, list[list[float]]] = {}
            for entry in entries:
                for label in self._entry_labels(entry, facet):
                    grouped.setdefault(label, []).append(entry_embeddings[entry.id])
            centroids[name] = {
                label: self._mean(vectors)
                for label, vectors in grouped.items()
            }

        self._entry_embeddings = entry_embeddings
        self._centroids = centroids
        self._source_hash = self._hash_entries(entries)
        self._persist()
        logger.info(
            "Centroid store rebuilt: %d entries, facets=%s",
            len(entries),
            {name: len(labels) for name, labels in centroids.items()},
        )

    # ------------------------------------------------------------ Hilfe

    @staticmethod
    def _entry_labels(entry: FAQEntry, facet: CentroidFacet) -> list[str]:
        """Labels an entry contributes for the facet.

        The facet's ``metadata_key`` names the FAQEntry attribute of the same
        name (e.g. ``question_type``, ``subjects``). Multi-label facets take
        every list element; single-label facets the stripped value.
        """
        value = getattr(entry, facet.metadata_key, "")
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        label = (value or "").strip()
        return [label] if label else []

    @staticmethod
    def _mean(vectors: list[list[float]]) -> list[float]:
        dim = len(vectors[0])
        sums = [0.0] * dim
        for vector in vectors:
            for i, value in enumerate(vector):
                sums[i] += value
        return [s / len(vectors) for s in sums]

    @staticmethod
    def _hash_entries(entries: list[FAQEntry]) -> str:
        """Stable hash across IDs — the IDs already carry the full content hash."""
        canonical = json.dumps(
            sorted(entry.id for entry in entries),
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA_VERSION,
            "embedding_model": self._model,
            "source_hash": self._source_hash,
            "entry_embeddings": self._entry_embeddings,
            "facets": self._centroids,
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
