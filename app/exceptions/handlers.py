import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.exceptions.base import FaqBotError
from app.exceptions.provider import (
    ProviderUnavailableError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI):

    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable_handler(
        request,
        exc: ProviderUnavailableError,
    ):

        logger.exception(exc.message)

        return JSONResponse(
            status_code=503,
            content={
                "detail": exc.message,
            },
        )

    @app.exception_handler(ProviderTimeoutError)
    async def provider_timeout_handler(
        request,
        exc: ProviderTimeoutError,
    ):

        logger.exception(exc.message)

        return JSONResponse(
            status_code=504,
            content={
                "detail": exc.message,
            },
        )

    @app.exception_handler(FaqBotError)
    async def application_error_handler(
        request,
        exc: FaqBotError,
    ):

        logger.exception(exc.message)

        return JSONResponse(
            status_code=500,
            content={
                "detail": exc.message,
            },
        )