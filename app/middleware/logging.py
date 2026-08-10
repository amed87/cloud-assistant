import logging
import time
import uuid

from fastapi import Request

logger = logging.getLogger(__name__)


async def logging_middleware(request: Request, call_next):

    request_id = str(uuid.uuid4())[:8]

    start = time.perf_counter()

    logger.info(
        "[%s] %s %s started",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

    except Exception:

        logger.exception(
            "[%s] Unhandled exception",
            request_id,
        )

        raise

    duration = time.perf_counter() - start

    logger.info(
        "[%s] %s completed in %.2fs (%d)",
        request_id,
        request.url.path,
        duration,
        response.status_code,
    )

    response.headers["X-Request-ID"] = request_id

    return response