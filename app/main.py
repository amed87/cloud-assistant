from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi import Request
from app.middleware.logging import logging_middleware
from app.middleware.profiling.profiler import profiler

from app.api.chat import router as chat_router
from app.config.settings import get_settings
from app.config.logging import configure_logging
from app.exceptions.handlers import register_exception_handlers
from app.api.teams import router as teams_router
from app.api.admin import router as admin_router

configure_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_measurement_started = False
    if settings.ENABLE_PROFILER and settings.PROFILER_RUN_ID:
        profiler.start_run_measurement(settings.PROFILER_RUN_ID)
        run_measurement_started = True

    try:
        yield
    finally:
        if run_measurement_started:
            profiler.stop_run_measurement()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

WEB_DIR = Path(__file__).parent / "web"
app.mount("/portal-assets", StaticFiles(directory=WEB_DIR), name="portal-assets")


@app.get("/", include_in_schema=False)
async def portal():
    return FileResponse(WEB_DIR / "index.html")

app.middleware("http")(logging_middleware)
app.middleware("http")(profiler.profile)

register_exception_handlers(app)

app.include_router(chat_router)

app.include_router(teams_router)
app.include_router(admin_router)