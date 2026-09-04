from fastapi import Request, Response
from app.config.settings import get_settings
from pandas import DataFrame as dt
from typing import Tuple, Callable
import time
import threading
import psutil
import os 
import json

# Dokumentiere die CPU-Last und RAM-Nutzung für eine spezifische Anfrage an den Bot
async def profiler_middleware(request: Request, call_next: Callable) -> Response:
    settings = get_settings()
    num_threads = settings.ollama_num_threads

    # Profile die Anfrage falls 'profile=true' und der Endpoint /chat ist
    if settings.ENABLE_PROFILER and request.method == 'POST' and str(request.url)[-5:] == "/chat":        
        try:
            content = await request.json()
        except Exception as e: # Logging oder Custom error einbauen!
            return await call_next(request)
        
        request._body = json.dumps(content).encode()
        response, metrics, duration = await profile(request, call_next, interval=settings.PROFILER_SAMPLE_RATE)
        stats = make_stats_table(metrics)
        write_profile_to_file(settings.PROFILER_OUTPUT_PATH, 'a', num_threads, metrics, stats, duration, content)
        return response
    
    # Andernfalls lasse es aus
    return await call_next(request)

# Sammle Daten zur Nutzung von CPU und RAM durch den Bot, Ollama und allgemein
async def profile(request: Request, call_next: Callable, interval: float=0.1) -> Tuple[Response, dt, float]:
    metrics = {
    "timestamps": [],
    "cpu_percentages_process": [],
    "cpu_percentages_total": [],
    "cpu_percentages_ollama": [],
    "memory_mb_process": [],
    "memory_mb_total": [],
    "memory_mb_ollama": []
    }
    process_info = get_process_info()
    stop_monitoring = threading.Event()

    # Erstelle den parallelen Thread fuer das Monitoring
    monitor_thread = threading.Thread(target=poll_metrics, args=(process_info, metrics, interval, stop_monitoring), daemon=True)

    # Führe die Anfrage aus
    monitor_thread.start()
    start_execution = time.time()
    try:
        result = await call_next(request)
    finally:
        # Stelle sicher, dass Monitoring auch dann stoppt, wenn call_next crasht.
        stop_monitoring.set()
        monitor_thread.join()

    # Sammle und formatiere die Ergebnisse
    total_duration = time.time() - start_execution
    table = make_profile_table(process_info, metrics)

    return result, table, total_duration

# Sammle Monitoring-Daten im Hintergrund
def poll_metrics(process_inf: dict, metrics: dict, interval: float, stop_monitoring: threading.Event):
    start_time = time.time()

    while not stop_monitoring.is_set():
        current_time = time.time() - start_time
        
        cpu = round((process_inf["process"].cpu_percent(interval=None) / process_inf["num_cores"]), 2)
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_ollama = list(round((p.cpu_percent(interval=None) / process_inf["num_cores"]), 2) for p in process_inf["ollama_processes"])
        mem = round((process_inf["process"].memory_info().rss / (1024 * 1024)), 2)
        mem_total = psutil.virtual_memory().percent
        mem_ollama = list(round((p.memory_info().rss / (1024*1024)), 2) for p in process_inf["ollama_processes"])

        metrics["timestamps"].append(round(current_time, 2))
        metrics["cpu_percentages_process"].append(cpu)
        metrics["cpu_percentages_total"].append(cpu_total)
        metrics["cpu_percentages_ollama"].append(cpu_ollama)
        metrics["memory_mb_process"].append(mem)
        metrics["memory_mb_total"].append(mem_total)
        metrics["memory_mb_ollama"].append(mem_ollama)

        time.sleep(interval)       

# Suche die Informationen über den Chatbot-Prozess und die Ollama-Prozesse heraus
def get_process_info() -> dict:
    process_info = {}
    process_info["pid"] = os.getpid()
    process_info["process"] = psutil.Process(process_info["pid"])
    process_info["ollama_process_ids"] = find_processes(['ollama', 'llama'])
    process_info["ollama_processes"] = [psutil.Process(p) for p in process_info["ollama_process_ids"]]
    process_info["ollama_names"] = [p.name() for p in process_info["ollama_processes"]]
    process_info["num_cores"] = psutil.cpu_count(logical=True)

    return process_info

# Erstelle eine Pandas-Dataframe mit den Resultaten des Monitoring
def make_profile_table(process_info: dict, metrics: dict) -> dt:
    cpu_metrics = {
        "time (s)": metrics["timestamps"],
        "CPU Chatbot (Prozent)": metrics["cpu_percentages_process"],
        "CPU total (Prozent)": metrics["cpu_percentages_total"],
    }

    mem_metrics = {
        "RAM Chatbot (MB)": metrics["memory_mb_process"],
        "RAM total (Prozent)": metrics ["memory_mb_total"]
    }

    ollama_table_cpu = dt(metrics["cpu_percentages_ollama"], columns=[f"CPU {proc} (Prozent)" for proc in process_info["ollama_names"]])
    ollama_table_mem = dt(metrics["memory_mb_ollama"], columns=[f"RAM {proc} (MB)" for proc in process_info["ollama_names"]])

    table = ollama_table_cpu.join(dt.from_dict(cpu_metrics)).join(ollama_table_mem).join(dt.from_dict(mem_metrics)).set_index("time (s)")

    return table

# Berechne statistische Werte und gebe sie als Pandas-Dataframe aus
def make_stats_table(metrics: dt) -> dt:
    means = metrics.mean(axis=0)
    maxs = metrics.max(axis=0)
    medians = metrics.median(axis=0)

    stats = dt(list(zip(means, medians, maxs)), columns=['Durchschnitt', 'Median', 'Maximum']).T
    stats.columns = metrics.columns
    return stats

# Benutze eine Liste von Schlüsselwörtern um nach spezifischen auf dem System laufenden Prozessen zu suchen.
def find_processes(keywords: list[str])-> list[str]: 
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
                for keyword in keywords:
                    if (keyword in cmd_string or keyword in proc.name().lower()) and proc.pid not in found_processes:
                        found_processes.append(proc.pid)
                    
        except psutil.AccessDenied:
            # Ignoriere Prozesse, für die dein Skript keine Rechte besitzt
            continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # Ignoriere Prozesse, die sich während der Schleife beendet haben
            continue
            
    return found_processes

# Schreibe die Resutate in ihre respektiven Dateien
def write_profile_to_file(path: str, mode: str, num_threads: str, metrics: dt, stats: dt, duration: float, content: any):
    with open(path, mode) as file:
        file.write(f"<br /><br />Zeit für Anfrage: {duration} Sekunden<br />")
        file.write(f"Threads: {num_threads}<br />")
        file.write(f"Conversation-ID: {content['conversation_id']}<br />")
        file.write(f"Anfrage: {content['message']}<br />")
        metrics.to_html(file)
        stats.to_html(file)

    # Schreibe die für die Anfrage benötigte Zeit in die korrespondierende CSV-Datei
    perf_csv = path[:-4] + "csv"
    perf_header = "Thread-Zahl, Zeit fuer Anfrage (s), Conversation-ID, Message\r\n"
    perf_str = f"{num_threads}, {duration}, {content['conversation_id']}, {content['message']} \r\n"
    with open(perf_csv, 'a', newline='', encoding='utf-8') as file:
        try:
            if os.path.getsize(perf_csv) == 0:
                file.write(perf_header )
            file.write(perf_str)
        except FileNotFoundError:
            pass
        