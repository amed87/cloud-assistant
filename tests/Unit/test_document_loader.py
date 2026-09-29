import pytest

from app.documents.faq_loader_csv import FAQLoaderCSV
from app.documents.models import FAQDocument, TextDocument
from app.documents.text_loader import TextLoader


@pytest.mark.asyncio
async def test_text_loader_loads_text_document(tmp_path) -> None:
    input_file = tmp_path / "document.txt"
    input_file.write_text("Example content", encoding="utf-8")

    document = await TextLoader().load(str(input_file))

    assert isinstance(document, TextDocument)
    assert document.id == "document"
    assert document.source == str(input_file)
    assert document.content == "Example content"


@pytest.mark.asyncio
async def test_faq_loader_csv_loads_faq_document(tmp_path) -> None:
    input_file = tmp_path / "faq.csv"
    input_file.write_text(
        "ID;Frage;Antwort;Themen;Fragetyp;Schlüsselwörter\n"
        "python;Was ist Python?;Eine Programmiersprache;Programmierung;Definition;python programmierung\n",
        encoding="utf-8",
    )

    document = await FAQLoaderCSV().load(str(input_file))

    assert isinstance(document, FAQDocument)
    assert document.id == "faq"
    assert document.source == str(input_file)
    assert len(document.entries) == 1

    entry = document.entries[0]
    assert entry.id.startswith("python:")
    assert entry.question == "Was ist Python?"
    assert entry.answer == "Eine Programmiersprache"
    assert entry.subjects == ["Programmierung"]
    assert entry.question_type == "Definition"
    assert entry.keywords == ["python", "programmierung"]


@pytest.mark.asyncio
async def test_text_loader_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        await TextLoader().load(str(tmp_path / "missing.txt"))


@pytest.mark.asyncio
async def test_faq_loader_csv_raises_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        await FAQLoaderCSV().load(str(tmp_path / "missing.csv"))