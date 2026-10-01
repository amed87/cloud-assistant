from dataclasses import dataclass
from fastapi import Request, Response
from pandas import DataFrame 
from typing import Callable
import time
import threading
import json
import logging
from app.middleware.profiling.monitor import ProcessMonitor
from app.middleware.profiling.reporting import Measurement, write_report
from app.config.settings import get_settings


logger = logging.getLogger(__name__)

@dataclass
class ProfileResult:
    response: Response
    metrics: DataFrame
    duration_seconds: float


class Profiler:
    """Hold process-wide profiler state for sampling and report serialization."""

    def __init__(self, profiled_endpoints: set[str]) -> None:
        settings = get_settings()
        self._lock = threading.Lock()
        self._monitor = ProcessMonitor()
        self._profiled_endpoints = profiled_endpoints
        self._sample_rate = settings.PROFILER_SAMPLE_RATE
        self._num_ollama_threads = settings.ollama_num_threads
        self._output_path = settings.PROFILER_OUTPUT_PATH
        self._output_format = settings.PROFILER_OUTPUT_FORMAT

    def _wants_profiling(self, request: Request) -> bool:
        return (
            get_settings().ENABLE_PROFILER
            and request.method == "POST"
            and request.url.path in self._profiled_endpoints
        )

    def _try_acquire(self) -> bool:
        """Acquire the profiler session if available; return False to skip profiling."""
        if self._lock.locked():
            return False
        self._lock.acquire()
        return True

    def _release(self) -> None:
        self._lock.release()

    async def profile(self, request: Request, call_next: Callable) -> Response:
        if not self._wants_profiling(request):
            return await call_next(request)

        free = self._try_acquire()
        if not free:
            logger.warning("Profiler is already active. Skipping profiling for this request.")
            return await call_next(request)

        try:
            try:
                content = await request.json()
            except Exception:
                logger.exception("Failed to parse profiling request JSON.")
                return await call_next(request)
            
            request._body = json.dumps(content).encode()
           
            start_execution = time.time()
            sampling = self._monitor.start_sampling(self._sample_rate)
            try:
                response = await call_next(request)
            finally:
                series = sampling.stop() 
            total_duration = time.time() - start_execution

            metrics_df = series.to_dataframe()
            measurement = Measurement(
                endpoint=request.url.path,
                num_threads=self._num_ollama_threads,
                duration_seconds=total_duration,
                conversation_id=content.get("conversation_id", "?"),
                message=content.get("message", "?"),
                metrics=metrics_df,
            )
            try:
                write_report(self._output_path, measurement)
            except Exception:
                logger.exception("Failed to write profiling report.")

            return response
        finally:
            self._release()


profiler = Profiler({"/chat", "/chat/quick"})