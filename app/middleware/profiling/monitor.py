import psutil
import os
import threading
import time
from dataclasses import dataclass, field
from pandas import DataFrame

@dataclass
class Sample:
    """Ein einzelner Messzeitpunkt."""
    cpu: float
    cpu_total: float
    cpu_ollama: list[float]
    mem: float
    mem_total: float
    mem_ollama: list[float]


@dataclass
class MetricsSeries:
    ollama_names: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    cpu_percentages_process: list[float] = field(default_factory=list)
    cpu_percentages_total: list[float] = field(default_factory=list)
    cpu_percentages_ollama: list[list[float]] = field(default_factory=list)
    memory_mb_process: list[float] = field(default_factory=list)
    memory_mb_total: list[float] = field(default_factory=list)
    memory_mb_ollama: list[list[float]] = field(default_factory=list)

    def append(self, timestamp: float, sample: Sample) -> None:
        """Hängt ein Sample an die Zeitreihen an."""
        self.timestamps.append(timestamp)
        self.cpu_percentages_process.append(sample.cpu)
        self.cpu_percentages_total.append(sample.cpu_total)
        self.cpu_percentages_ollama.append(sample.cpu_ollama)
        self.memory_mb_process.append(sample.mem)
        self.memory_mb_total.append(sample.mem_total)
        self.memory_mb_ollama.append(sample.mem_ollama)

    def to_dataframe(self) -> DataFrame:
        """Baut die Tabelle für den HTML-Report."""
        table = DataFrame({
            "time (s)": self.timestamps,
            "CPU Chatbot (Prozent)": self.cpu_percentages_process,
            "CPU total (Prozent)": self.cpu_percentages_total,
            "RAM Chatbot (MB)": self.memory_mb_process,
            "RAM total (Prozent)": self.memory_mb_total,
        })

        for i, name in enumerate(self.ollama_names):
            table[f"CPU {name} (Prozent)"] = [row[i] for row in self.cpu_percentages_ollama]
            table[f"RAM {name} (MB)"] = [row[i] for row in self.memory_mb_ollama]

        return table.set_index("time (s)")

class SamplingSession:
    """Ein laufender Messlauf; stop() beendet ihn und liefert die Series."""

    def __init__(self, series: MetricsSeries, thread: threading.Thread, stop_event: threading.Event) -> None:
        self._series = series
        self._thread = thread
        self._stop = stop_event

    def stop(self) -> MetricsSeries:
        self._stop.set()
        self._thread.join()
        return self._series


# process discovery + sampling
class ProcessMonitor:
    """Überwacht CPU/RAM des Chatbot- und Ollama-Prozesse."""

    def __init__(self) -> None:
        self._info: dict | None = None
        self._keywords: list[str] = ["ollama", "llama"]

    def start_sampling(self, interval: float) -> SamplingSession:
        """Startet das Hintergrund-Sampling. Ruft bei Bedarf discover() auf."""
        info = self._discover()                                    
        series = MetricsSeries(ollama_names=info["ollama_names"])
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._poll_loop,
            args=(series, interval, stop_event),
            daemon=True,
        )
        thread.start()
        return SamplingSession(series, thread, stop_event) 

    def _discover(self, refresh: bool = False) -> dict:
        """Findet die Prozesse einmalig und cached sie (refresh=False-Default)."""
        if self._info is not None and not refresh:
            return self._info
        
        process_info = {}
        process_info["pid"] = os.getpid()
        process_info["process"] = psutil.Process(process_info["pid"])
        process_info["ollama_process_ids"] = self._find_processes()
        process_info["ollama_processes"] = [psutil.Process(p) for p in process_info["ollama_process_ids"]]
        process_info["ollama_names"] = [p.name() for p in process_info["ollama_processes"]]
        process_info["num_cores"] = psutil.cpu_count(logical=True)

        if process_info["ollama_processes"]:
            self._info = process_info
        return process_info

    # Durchsucht alle Systemprozesse nach den Keywords in self._keywords und liefert deren PIDs.
    def _find_processes(self)-> list[int]: 
        found_processes = []        
        # Durchlaufe alle aktiven Systemprozesse
        for proc in psutil.process_iter():
            try:
                # Hole die vollständigen Kommandozeilen-Argumente als Liste
                cmdline_list = proc.cmdline()                
                # Falls die Liste existiert, wandle sie in einen String um
                if cmdline_list:
                    cmd_string = " ".join(cmdline_list).lower()                    
                    # Prüfe, ob 'ollama' im Befehl vorkommt
                    for keyword in self._keywords:
                        if (keyword in cmd_string or keyword in proc.name().lower()) and proc.pid not in found_processes:
                            found_processes.append(proc.pid)
                        
            except psutil.AccessDenied:
                # Ignoriere Prozesse, für die dein Skript keine Rechte besitzt
                continue
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # Ignoriere Prozesse, die sich während der Schleife beendet haben
                continue
                
        return found_processes

    def _warmup(self) -> None:
        info = self._discover()
        info["process"].cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        for p in info["ollama_processes"]:
            p.cpu_percent(interval=None)

    def _collect_sample(self) -> Sample:
        """Ein einzelnes Sample – wirft RuntimeError wenn ProcessMonitor._discover noch nicht gelaufen ist und psutil.NoSuchProcess bei toten Handles."""
        info = self._info
        if info is None:
            raise RuntimeError(
                "ProcessMonitor.discover() wurde nicht aufgerufen."
            )
        return Sample(
            cpu = round(info["process"].cpu_percent(interval=None) / info["num_cores"], 2),
            cpu_total = psutil.cpu_percent(interval=None),
            cpu_ollama = [round(p.cpu_percent(interval=None) / info["num_cores"], 2) for p in info["ollama_processes"]],
            mem = round(info["process"].memory_info().rss / (1024*1024), 2),
            mem_total = psutil.virtual_memory().percent,
            mem_ollama = [p.memory_info().rss / (1024*1024) for p in info["ollama_processes"]],
        )

    def _refresh(self) -> None:
        self._discover(refresh=True)
        self._warmup()

    

    def _poll_loop(self, series: MetricsSeries, interval: float, stop_event: threading.Event) -> None:
        self._warmup()
        start_time = time.time()
        while not stop_event.is_set():
            try:
                sample = self._collect_sample()
            except psutil.NoSuchProcess:
                self._refresh()
                continue
            series.append(round(time.time() - start_time, 2), sample)
            time.sleep(interval)