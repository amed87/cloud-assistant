"""Exercise temporary Chroma storage operations and backend error translation."""

import pytest

from app.exceptions.vectorstore import VectorStoreError
from app.vectorstores.models import VectorDocument
from app.vectorstores.temp_chroma import TempChromaStore


@pytest.mark.asyncio
async def test_temp_chroma_add_search_get_delete_and_close(tmp_path) -> None:
    store_path = tmp_path / "temporary-chroma"
    store = TempChromaStore(str(store_path))
    document = VectorDocument(
        id="faq-1",
        content="Question and answer",
        embedding=[1.0, 0.0, 0.0],
        metadata={"answer": "Answer"},
    )

    await store.add([document])
    results = await store.search([1.0, 0.0, 0.0], limit=2)
    database = store.get()

    assert len(results) == 1
    assert results[0].id == document.id
    assert results[0].content == document.content
    assert results[0].score == pytest.approx(1.0)
    assert results[0].metadata == document.metadata
    assert database["ids"] == [document.id]

    await store.delete([document.id])

    assert store.get()["ids"] == []

    await store.close()


@pytest.mark.asyncio
async def test_temp_chroma_wraps_backend_errors_as_vector_store_errors(
    tmp_path,
) -> None:
    store = TempChromaStore(str(tmp_path / "temporary-chroma-errors"))
    error = RuntimeError("backend detail")

    def fail_upsert(**kwargs) -> None:
        raise error

    store.collection.upsert = fail_upsert

    with pytest.raises(VectorStoreError) as raised:
        await store.add(
            [
                VectorDocument(
                    id="faq-1",
                    content="Question",
                    embedding=[1.0, 0.0],
                )
            ]
        )

    assert raised.value.__cause__ is error
    await store.close()
