"""Ergebnis-Aufbereitung und Persistenz der Profiler-Messungen."""

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
    """Alle Daten einer einzelnen profilierten Anfrage."""
    endpoint: str
    num_threads: str
    duration_seconds: float
    conversation_id: str
    message: str
    metrics: "DataFrame"     # Rohwerte (Zeitreihe)
    stats: "DataFrame | None" = None      # Durchschnitt/Median/Maximum je Spalte, von write_report gefüllt


def _make_stats_table(metrics: "DataFrame") -> "DataFrame":
    """Berechnet Durchschnitt, Median und Maximum je Messspalte."""
    means = metrics.mean(axis=0)
    maxs = metrics.max(axis=0)
    medians = metrics.median(axis=0)

    stats = DataFrame(
        list(zip(means, medians, maxs)),
        columns=["Durchschnitt", "Median", "Maximum"],
    ).T
    stats.columns = metrics.columns
    return stats


def write_report(path: str, measurement: Measurement) -> None:
    """Schreibt eine Messung als HTML-Bericht und CSV-Zeile."""
    if measurement.metrics.empty:
        logger.warning("Profiler: Leere Messung an Report-Writer übergeben.")
    if measurement.stats is None:
        measurement.stats = _make_stats_table(measurement.metrics)
    _write_html(path, measurement)
    _write_csv_row(path, measurement)


def _write_html(path: str, m: Measurement) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(f"<br /><br />Zeit für Anfrage: {m.duration_seconds} Sekunden<br />")
        file.write(f"Threads: {m.num_threads}<br />")
        file.write(f"Endpoint: {m.endpoint}<br />")
        file.write(f"Conversation-ID: {m.conversation_id}<br />")
        file.write(f"Anfrage: {m.message}<br />")
        m.metrics.to_html(file)
        m.stats.to_html(file)


def _write_csv_row(path: str, m: Measurement) -> None:
    csv_path = str(Path(path).with_suffix(".csv"))

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow([m.num_threads, m.duration_seconds, m.conversation_id, m.message, m.endpoint])
    row = buffer.getvalue()

    header = "Thread-Zahl,Zeit fuer Anfrage (s),Conversation-ID,Message,Endpoint\r\n"

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        try:
            if os.path.getsize(csv_path) == 0:
                file.write(header)
            file.write(row)
        except OSError as e:
            logger.error("Profiler konnte Messung nicht schreiben: %s", e)