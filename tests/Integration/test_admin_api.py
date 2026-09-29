from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.admin import router
from app.dependencies.db_update import get_update_service


class FakeUpdateService:
    async def update_database(self) -> None:
        pass


def create_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_update_service] = FakeUpdateService

    return TestClient(app)


def test_update_database_is_initiated() -> None:
    response = create_client().post("/admin/update-db")

    assert response.status_code == 200
    assert response.json() == {
        "status": "Database update initiated",
    }