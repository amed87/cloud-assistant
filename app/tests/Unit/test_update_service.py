import pytest

from app.services.update_service import UpdateService


class MockLoader:
    def __init__(self, document: object) -> None:
        self.document = document
        self.loaded_path: str | None = None

    async def load(self, path: str) -> object:
        self.loaded_path = path
        return self.document


class MockImporter:
    def __init__(self) -> None:
        self.imported_document: object | None = None

    async def import_data(self, document: object) -> None:
        self.imported_document = document


@pytest.mark.asyncio
async def test_update_database_loads_and_imports_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = object()
    loader = MockLoader(document)
    importer = MockImporter()

    settings = type(
        "Settings",
        (),
        {"FAQ_INPUT_FILE": "faq.csv"},
    )

    monkeypatch.setattr(
        "app.services.update_service.get_settings",
        lambda: settings(),
    )

    service = UpdateService(
        loader=loader,
        importer=importer,
    )

    await service.update_database()

    assert loader.loaded_path == "faq.csv"
    assert importer.imported_document is document