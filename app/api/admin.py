from fastapi import APIRouter, BackgroundTasks, Depends
from app.services.update_service import UpdateService
from app.dependencies.db_update import get_update_service

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],)

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/update-db")
async def update_database(background_tasks: BackgroundTasks, update_service: UpdateService = Depends(get_update_service)):
    try:
        background_tasks.add_task(update_service.update_database)
        return {"status": "Database update initiated"}

    except Exception as e:
        return {"status": f"Error occurred while initiating database update."}

@router.get("/db")
async def get_database(update_service: UpdateService = Depends(get_update_service)):
    try:
        db = update_service.get_database()
        return {"DB": db}
    except Exception as e:
        return {"status": "Error while fetching database"}
