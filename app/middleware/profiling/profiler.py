from fastapi import Request, Response
from typing import Callable
import logging
import time
import threading
import uuid

from app.config.settings import get_settings
from app.middleware.profiling.monitor import ProcessMonitor, SamplingSession
from app.middleware.profiling.reporting import (
    RequestMeasurement,
    RunResourceCsvWriter,
    write_request_measurement,
)


logger = logging.getLogger(__name__)

class Profiler:
    """Collect lightweight per-request performance measurements."""

    def __init__(self, profiled_endpoints: set[str]) -> None:
        settings = get_settings()
        self._enabled = settings.ENABLE_PROFILER
        self._profiled_endpoints = profiled_endpoints
        self._output_path = settings.PROFILER_OUTPUT_PATH
        self._monitor = ProcessMonitor()
        self._sample_interval = settings.PROFILER_SAMPLE_RATE
        self._run_lock = threading.Lock()
        self._run_id: str | None = None
        self._run_session: SamplingSession | None = None
        self._run_writer: RunResourceCsvWriter | None = None

    def start_run_measurement(self, run_id: str) -> None:
        """Start one resource-sampling session for a complete load-test run."""
        if not self._enabled:
            return
        if not run_id.strip():
            raise ValueError("run_id must not be empty")

        with self._run_lock:
            if self._run_session is not None:
                raise RuntimeError(
                    f"Resource sampling is already active for run {self._run_id}."
                )

            writer = RunResourceCsvWriter(self._output_path, run_id)
            writer.start()
            self._run_id = run_id
            self._run_writer = writer
            try:
                self._run_session = self._monitor.start_sampling(
                    self._sample_interval,
                    on_sample=writer.write_sample,
                    retain_samples=False,
                )
            except Exception:
                writer.stop()
                self._run_id = None
                self._run_writer = None
                raise
            logger.info("Started run-wide resource sampling for run %s.", run_id)

    def stop_run_measurement(self) -> None:
        """Stop the active run sampler and persist its resource samples."""
        with self._run_lock:
            if self._run_session is None or self._run_id is None:
                return

            run_id = self._run_id
            session = self._run_session
            writer = self._run_writer
            try:
                session.stop()
            finally:
                self._run_session = None
                self._run_id = None
                self._run_writer = None
                if writer is not None:
                    try:
                        writer.stop()
                    except Exception:
                        logger.exception(
                            "Failed to close resource CSV for run %s.",
                            run_id,
                        )

    def _wants_profiling(self, request: Request) -> bool:
        return (
            self._enabled
            and request.method == "POST"
            and request.url.path in self._profiled_endpoints
        )

    async def profile(self, request: Request, call_next: Callable) -> Response:
        if not self._wants_profiling(request):
            return await call_next(request)

        start_time = time.perf_counter()
        with self._run_lock:
            active_run_id = self._run_id
        run_id = request.headers.get("X-Run-ID") or active_run_id or ""
        case_id = request.headers.get("X-Case-ID")
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        status_code = 500
        error_type = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            request_id = response.headers.get("X-Request-ID", request_id)
            return response
        except Exception as ex:
            error_type = type(ex).__name__
            raise
        finally:
            measurement = RequestMeasurement(
                run_id=run_id,
                request_id=request_id,
                case_id=case_id,
                endpoint=request.url.path,
                status_code=status_code,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error_type=error_type,
            )
            try:
                write_request_measurement(self._output_path, measurement)
            except Exception:
                logger.exception("Failed to write request profiling data.")


profiler = Profiler({"/chat", "/chat/quick"})