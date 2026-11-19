r"""End-to-End: Der DB-Update-Lauf befüllt Chroma und die Zentroide gemeinsam.

Kette wie in Produktion (app.dependencies.db_update/centroids), nur mit
deterministischem Fake-Embedding statt Ollama:

    FAQLoaderCSV -> FAQImporter -> ChromaVectorStore (tmp-DB)
 `-> CentroidStore.rebuild -> centroids.json

Geprüft wird das Zusammenspiel: Einträge landen in der Vektordatenbank, die
Zentroide spiegeln die finalen Einträge, mehrere Updates pushen nur Deltas
(neue/ geänderte Fragen werden neu eingebettet, alte IDs räumen sich weg).
"""

import json
from pathlib import Path

import pytest

from app.config.content_settings import CentroidFacet
from app.config.settings import get_settings
from app.documents.faq_importer import FAQImporter
from app.documents.faq_loader_csv import FAQLoaderCSV
from app.quality.classification.centroid_store import CentroidStore
from app.services.update_service import UpdateService
from app.vectorstores.chroma import ChromaVectorStore


# ------------------------------------------------------------------ Helfer

class FakeEmbeddingProvider:
    """Deterministic embeddings; the question's last char drives the vector."""

    model = "fake-embed"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        return list(vectors[text[-1]])


FACETS = {
    "fragetyp": CentroidFacet(metadata_key="question_type", fallback="Generell"),
    "thema": CentroidFacet(
        metadata_key="subjects", multi_label=True, fallback="Allgemein",
    ),
}

CSV_HEADER = "ID;Frage;Antwort;Themen;Fragetyp;Schlüsselwörter"


def write_source_csv(path: Path, rows: list[str]) -> None:
    path.write_text(
        "\n".join([CSV_HEADER, *rows]) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------- Tests

@pytest.mark.asyncio
async def test_update_writes_chroma_and_centroids_and_deltas_are_incremental(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embedding = FakeEmbeddingProvider()
    source_csv = tmp_path / "faq.csv"
    db_path = tmp_path / "chroma"
    centroid_path = db_path / "centroids.json"

    settings = get_settings()
    monkeypatch.setattr(settings, "DB_PATH", str(db_path))
    monkeypatch.setattr(settings, "CHROMA_COLLECTION", "update-centroids-test")
    monkeypatch.setattr(settings, "FAQ_INPUT_FILE", str(source_csv))

    # Erster Lauf: zwei Einträge.
    write_source_csv(
        source_csv,
        [
            "1;frage a;answer a1;ThemaA;Wie;",
            "2;frage b;answer b1;ThemaA ThemaB;Was;",
        ],
    )

    vector_store = ChromaVectorStore()
    centroid_store = CentroidStore(
        store_path=centroid_path,
        embedding_provider=fake_embedding,
        facets=FACETS,
    )
    service = UpdateService(
        loader=FAQLoaderCSV(),
        importer=FAQImporter(
            embedding_provider=fake_embedding,
            vector_store=vector_store,
        ),
        centroid_store=centroid_store,
    )

    await service.update_database()

    # Chroma enthält beide Fragen (IDs tragen den Content-Hash).
    database = vector_store.get()
    assert len(database["ids"]) == 2
    metadata_by_id = dict(zip(database["ids"], database["metadatas"]))
    assert all(
        meta["question_type"] in {"Wie", "Was"}
        and meta["source"].endswith("faq.csv")
        for meta in metadata_by_id.values()
    )

    # Die Zentroid-Datei liegt neben der Chroma-DB und trägt die Facetten.
    assert centroid_path.is_file()
    payload = json.loads(centroid_path.read_text(encoding="utf-8"))
    assert payload["schema"] == 1
    assert payload["embedding_model"] == "fake-embed"
    assert set(payload["facets"]) == {"fragetyp", "thema"}
    # Deterministische Frage-Embeddings: 'frage a' -> [1, 0], 'frage b' -> [0, 1].
    assert payload["facets"]["fragetyp"]["Wie"] == [1.0, 0.0]
    assert payload["facets"]["fragetyp"]["Was"] == [0.0, 1.0]
    assert payload["facets"]["thema"]["ThemaA"] == pytest.approx([0.5, 0.5])
    assert payload["facets"]["thema"]["ThemaB"] == [0.0, 1.0]

    # Embedding-Aufrufe: je Komponente (Importer, Store) alle 2 neuen Fragen.
    assert fake_embedding.calls == ["frage a", "frage b"] * 2

    # Zweiter Lauf: Antwortänderung (neue ID), neuer Eintrag, Frage 2 bleibt.
    # Nur die zwei betroffenen Fragen dürfen je Komponente neu eingebettet
    # werden — der Embedding-Cache im Store überlebt die unveränderte Frage.
    write_source_csv(
        source_csv,
        [
            "1;frage a;answer a2;ThemaA;Wie;",
            "2;frage b;answer b1;ThemaA ThemaB;Was;",
            "3;frage a;answer c1;ThemaB;Was;",
        ],
    )

    first_run_ids = set(database["ids"])
    old_id_1 = next(
        entry_id
        for entry_id, meta in metadata_by_id.items()
        if meta["answer"] == "answer a1"
    )

    fake_embedding.calls.clear()
    await service.update_database()

    database = vector_store.get()
    new_ids = set(database["ids"])
    new_metadata = dict(zip(new_ids, database["metadatas"]))

    # Der geänderte Eintrag 1 wurde synchronisiert (alte ID weg, neue da).
    assert old_id_1 not in new_ids
    assert len(new_ids - first_run_ids) == 2  # geänderte Frage 1 + neue Frage 3
    assert sum(meta["answer"] == "answer a2" for meta in new_metadata.values()) == 1
    assert sum(meta["answer"] == "answer b1" for meta in new_metadata.values()) == 1

    # Nur Frage 1 (geändert, 'frage a') und Frage 3 (neu, 'frage a') wurden
    # je Komponente gebettet; Frage 2 blieb kalt (2 x 2 Aufrufe von 'frage a').
    assert fake_embedding.calls == ["frage a"] * 4

    # Zentroide spiegeln den Endstand inkl. 'Was'-Mischung aus b/a-Vektoren.
    reloaded_payload = json.loads(centroid_path.read_text(encoding="utf-8"))
    assert reloaded_payload["facets"]["fragetyp"]["Was"] == pytest.approx([0.5, 0.5])
    assert reloaded_payload["facets"]["fragetyp"]["Wie"] == [1.0, 0.0]
    assert reloaded_payload["facets"]["thema"]["ThemaA"] == pytest.approx([0.5, 0.5])
    assert reloaded_payload["facets"]["thema"]["ThemaB"] == pytest.approx([0.5, 0.5])

    # Alter Content-Hash räumt sich aus dem Embedding-Cache weg.
    assert set(reloaded_payload["entry_embeddings"]) == new_ids

    # Eine frische Store-Instanz (neuer Prozess) lädt den Endstand sauber.
    fresh_store = CentroidStore(
        store_path=centroid_path,
        embedding_provider=fake_embedding,
        facets=FACETS,
    )
    assert fresh_store.load() is True
    assert set(fresh_store.labels("fragetyp")) == {"Wie", "Was"}
    assert fresh_store.labels("fragetyp")["Was"] == pytest.approx([0.5, 0.5])
