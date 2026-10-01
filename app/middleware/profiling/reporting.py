"""Prepare and persist profiler measurements."""

import csv
import io
import logging
import os
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestMeasurement:
    """Performance data for one HTTP request."""
    run_id: str
    request_id: str
    case_id: str | None
    endpoint: str
    status_code: int
    duration_ms: float
    error_type: str | None = None


def write_request_measurement(
    path: str,
    measurement: RequestMeasurement,
) -> None:
    """Append one request measurement to the CSV file."""
    csv_path = str(Path(path).with_suffix(".csv"))
    header = [
        "Run ID",
        "Request ID",
        "Case ID",
        "Endpoint",
        "Status Code",
        "Duration (ms)",
        "Error Type",
    ]
    row = [
        measurement.run_id,
        measurement.request_id,
        measurement.case_id or "",
        measurement.endpoint,
        measurement.status_code,
        measurement.duration_ms,
        measurement.error_type or "",
    ]
    _append_csv_row(csv_path, header, row)


class RunResourceCsvWriter:
    """Buffer run resource rows and periodically flush them to an open CSV stream."""

    def __init__(
        self,
        path: str,
        run_id: str,
        flush_interval: float = 1.0,
    ) -> None:
        if flush_interval <= 0:
            raise ValueError("flush_interval must be greater than zero")

        output_path = Path(path)
        self._csv_path = output_path.with_name(f"{output_path.stem}-resources.csv")
        self.run_id = run_id
        self.flush_interval = flush_interval
        self._file = None
        self._writer = None
        self._pending_rows: list[list[object]] = []
        self._last_flush = 0.0

    def start(self) -> None:
        """Open the resource CSV stream and write its header if needed."""
        if self._file is not None:
            raise RuntimeError("Resource CSV writer is already started.")

        self._file = self._csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, lineterminator="\r\n")
        if self._csv_path.stat().st_size == 0:
            self._writer.writerow([
                "Run ID",
                "Elapsed Time (s)",
                "Resource",
                "CPU (%)",
                "Memory (MB)",
                "Memory (%)",
            ])
            self._flush()

    def write_sample(
        self,
        elapsed_time: float,
        metrics: Mapping[str, object],
    ) -> None:
        """Queue the resource rows for one sampling instant."""
        if self._file is None or self._writer is None:
            raise RuntimeError("Resource CSV writer has not been started.")
        self._pending_rows.extend(
            _resource_rows(self.run_id, elapsed_time, metrics)
        )
        if time.monotonic() - self._last_flush >= self.flush_interval:
            self._flush()

    def flush(self) -> None:
        """Flush buffered rows to the CSV file."""
        self._flush()

    def stop(self) -> None:
        """Flush pending rows and close the CSV stream."""
        if self._file is None:
            return
        try:
            self._flush()
        finally:
            self._file.close()
            self._file = None
            self._writer = None

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    def _flush(self) -> None:
        if self._file is None or self._writer is None:
            raise RuntimeError("Resource CSV writer has not been started.")
        if self._pending_rows:
            self._writer.writerows(self._pending_rows)
            self._pending_rows.clear()
        self._file.flush()
        self._last_flush = time.monotonic()

    def __enter__(self) -> "RunResourceCsvWriter":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def _resource_rows(
    run_id: str,
    elapsed_time: float,
    metrics: Mapping[str, object],
) -> list[list[object]]:
    rows = [[
        run_id,
        elapsed_time,
        "chatbot",
        metrics.get("Chatbot CPU (%)"),
        metrics.get("Chatbot RAM (MB)"),
        None,
    ], [
        run_id,
        elapsed_time,
        "system",
        metrics.get("Total CPU (%)"),
        None,
        metrics.get("Total RAM (%)"),
    ]]

    process_names = {
        column.removeprefix("CPU ").removesuffix(" (%)")
        for column in metrics
        if column.startswith("CPU ") and column.endswith(" (%)")
    }
    for process_name in sorted(process_names - {"Chatbot"}):
        rows.append([
            run_id,
            elapsed_time,
            f"ollama:{process_name}",
            metrics.get(f"CPU {process_name} (%)"),
            metrics.get(f"RAM {process_name} (MB)"),
            None,
        ])
    return rows


def _append_csv_row(
    csv_path: str,
    header: list[str] | str,
    row: list[object],
) -> None:
    _append_csv_rows(csv_path, header, [row])


def _append_csv_rows(
    csv_path: str,
    header: list[str] | str,
    rows: list[list[object]],
) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        try:
            if os.path.getsize(csv_path) == 0:
                if isinstance(header, str):
                    file.write(header)
                else:
                    csv.writer(file, lineterminator="\r\n").writerow(header)
            csv.writer(file, lineterminator="\r\n").writerows(rows)
        except OSError as e:
            logger.error("Failed to write profiling metrics: %s", e)