import psutil
import os
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pandas import DataFrame

logger = logging.getLogger(__name__)

@dataclass
class Sample:
    """A single measurement sample."""
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
        """Append a sample to the time series."""
        self.timestamps.append(timestamp)
        self.cpu_percentages_process.append(sample.cpu)
        self.cpu_percentages_total.append(sample.cpu_total)
        self.cpu_percentages_ollama.append(sample.cpu_ollama)
        self.memory_mb_process.append(sample.mem)
        self.memory_mb_total.append(sample.mem_total)
        self.memory_mb_ollama.append(sample.mem_ollama)

    def to_dataframe(self) -> DataFrame:
        """Build the table used in the HTML report."""
        table = DataFrame({
            "time (s)": self.timestamps,
            "Chatbot CPU (%)": self.cpu_percentages_process,
            "Total CPU (%)": self.cpu_percentages_total,
            "Chatbot RAM (MB)": self.memory_mb_process,
            "Total RAM (%)": self.memory_mb_total,
        })

        for i, name in enumerate(self.ollama_names):
            table[f"CPU {name} (%)"] = [row[i] for row in self.cpu_percentages_ollama]
            table[f"RAM {name} (MB)"] = [row[i] for row in self.memory_mb_ollama]

        return table.set_index("time (s)")

class SamplingSession:
    """A running sampling session; stop() ends it and returns the series."""

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
    """Monitor CPU and memory usage for the chatbot and Ollama processes."""

    def __init__(self) -> None:
        self._info: dict | None = None
        self._keywords: list[str] = ["ollama", "llama"]

    def start_sampling(
        self,
        interval: float,
        on_sample: Callable[[float, dict[str, float]], None] | None = None,
        retain_samples: bool = True,
    ) -> SamplingSession:
        """Start background sampling, discovering processes if needed."""
        info = self._discover()                                    
        series = MetricsSeries(ollama_names=info["ollama_names"])
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._poll_loop,
            args=(series, interval, stop_event, on_sample, retain_samples),
            daemon=True,
        )
        thread.start()
        return SamplingSession(series, thread, stop_event) 

    def _discover(self, refresh: bool = False) -> dict:
        """Discover processes once and cache them unless refresh is requested."""
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

    # Find all system processes whose names or command lines match self._keywords.
    def _find_processes(self)-> list[int]: 
        found_processes = []        
        # Iterate over all active system processes.
        for proc in psutil.process_iter():
            try:
                # Get the full command line as a list.
                cmdline_list = proc.cmdline()                
                # Join the command-line arguments when available.
                if cmdline_list:
                    cmd_string = " ".join(cmdline_list).lower()                    
                    # Check whether a configured keyword matches the command.
                    for keyword in self._keywords:
                        if (keyword in cmd_string or keyword in proc.name().lower()) and proc.pid not in found_processes:
                            found_processes.append(proc.pid)
                        
            except psutil.AccessDenied:
                # Skip processes that cannot be accessed.
                continue
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # Skip processes that exited while iterating.
                continue
                
        return found_processes

    def _warmup(self) -> None:
        info = self._discover()
        info["process"].cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        for p in info["ollama_processes"]:
            p.cpu_percent(interval=None)

    def _collect_sample(self) -> Sample:
        """Collect one sample; may raise RuntimeError before discovery or NoSuchProcess for stale handles."""
        info = self._info
        if info is None:
            raise RuntimeError(
                "ProcessMonitor.discover() has not been called."
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

    

    def _poll_loop(
        self,
        series: MetricsSeries,
        interval: float,
        stop_event: threading.Event,
        on_sample: Callable[[float, dict[str, float]], None] | None = None,
        retain_samples: bool = True,
    ) -> None:
        self._warmup()
        start_time = time.perf_counter()
        while not stop_event.is_set():
            try:
                sample = self._collect_sample()
            except psutil.NoSuchProcess:
                self._refresh()
                continue
            elapsed_time = round(time.perf_counter() - start_time, 2)
            if retain_samples:
                series.append(elapsed_time, sample)
            if on_sample is not None:
                resource_metrics = {
                    "Chatbot CPU (%)": sample.cpu,
                    "Total CPU (%)": sample.cpu_total,
                    "Chatbot RAM (MB)": sample.mem,
                    "Total RAM (%)": sample.mem_total,
                }
                for index, name in enumerate(series.ollama_names):
                    resource_metrics[f"CPU {name} (%)"] = sample.cpu_ollama[index]
                    resource_metrics[f"RAM {name} (MB)"] = sample.mem_ollama[index]
                try:
                    on_sample(elapsed_time, resource_metrics)
                except Exception:
                    logger.exception(
                        "Resource sample callback failed; disabling the callback."
                    )
                    on_sample = None
            time.sleep(interval)