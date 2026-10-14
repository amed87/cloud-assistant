import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.services.update_service import UpdateService
from app.tools.faq_health_check import FAQHealthCheckService
from app.models.faq_health import FAQHealthCheckRequest, FAQHealthCheckReport
from app.dependencies.db_update import get_update_service
from app.dependencies.faq_health_check import get_faq_health_check_service
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.post("/update-db")
async def update_database(
    background_tasks: BackgroundTasks,
    update_service: UpdateService = Depends(get_update_service),
):
    background_tasks.add_task(update_service.update_database)
    return {"status": "Database update initiated"}


@router.get("/db")
async def get_database(update_service: UpdateService = Depends(get_update_service)):
    db = update_service.get_database()
    return {"DB": db}


@router.post(
    "/faq-health-check",
    response_model=FAQHealthCheckReport,
    summary="FAQ Health Check",
    description="Führt vollständigen QA-Check auf neuen FAQs durch. Läuft in isolierter temporärer ChromaDB.",
)
@router.post(
    "/faq-health-check",
    response_model=FAQHealthCheckReport,
)
async def faq_health_check(
    request: FAQHealthCheckRequest,
    qa_service: FAQHealthCheckService = Depends(get_faq_health_check_service),
):
    """Führt vollständigen QA-Check auf neuen FAQs durch."""
    try:
        file_path = request.file_path or get_settings().FAQ_INPUT_FILE
        logger.info("Starting FAQ health check for: %s", file_path)

        # Service macht ALLES (Load → TempStore → Import → Checks)
        report = await qa_service.run_qa(file_path)

        return report

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    except Exception as ex:
        logger.exception("FAQ Health Check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(ex)}")