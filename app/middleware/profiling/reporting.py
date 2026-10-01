"""Prepare and persist profiler measurements."""

import csv
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from pandas import DataFrame

logger = logging.getLogger(__name__)


@dataclass
class Measurement:
    """Data collected for a single profiled request."""
    endpoint: str
    num_threads: str
    duration_seconds: float
    conversation_id: str
    message: str
    metrics: "DataFrame"     # Raw time-series measurements.
    stats: "DataFrame | None" = None      # Per-column mean/median/maximum, populated by write_report.


def _make_stats_table(metrics: "DataFrame") -> "DataFrame":
    """Calculate the mean, median, and maximum for each measurement column."""
    means = metrics.mean(axis=0)
    maxs = metrics.max(axis=0)
    medians = metrics.median(axis=0)

    stats = DataFrame(
        list(zip(means, medians, maxs)),
        columns=["Mean", "Median", "Maximum"],
    ).T
    stats.columns = metrics.columns
    return stats


def write_report(path: str, measurement: Measurement) -> None:
    """Write a measurement to the HTML report and CSV file."""
    if measurement.metrics.empty:
        logger.warning("Profiler received an empty measurement series.")
    if measurement.stats is None:
        measurement.stats = _make_stats_table(measurement.metrics)
    _write_html(path, measurement)
    _write_csv_row(path, measurement)


def _write_html(path: str, m: Measurement) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(f"<br /><br />Request duration: {m.duration_seconds} seconds<br />")
        file.write(f"Thread count: {m.num_threads}<br />")
        file.write(f"Endpoint: {m.endpoint}<br />")
        file.write(f"Conversation ID: {m.conversation_id}<br />")
        file.write(f"Request: {m.message}<br />")
        m.metrics.to_html(file)
        m.stats.to_html(file)


def _write_csv_row(path: str, m: Measurement) -> None:
    csv_path = str(Path(path).with_suffix(".csv"))

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow([m.num_threads, m.duration_seconds, m.conversation_id, m.message, m.endpoint])
    row = buffer.getvalue()

    header = "Thread count,Request duration (s),Conversation ID,Message,Endpoint\r\n"

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        try:
            if os.path.getsize(csv_path) == 0:
                file.write(header)
            file.write(row)
        except OSError as e:
            logger.error("Failed to write profiling metrics: %s", e)