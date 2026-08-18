from fastapi import Request
from app.config.settings import get_settings
from pandas import DataFrame as dt
import time
import threading
import psutil
import os 

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

# Sammle Daten zur Nutzung von CPU und RAM durch den Bot, Ollama und allgemein
async def profile(request: Request, call_next, interval=0.1):
    print("Profiling")
    pid = os.getpid()
    process = psutil.Process(pid)
    ollama_process_ids = find_processes(['ollama', 'llama'])
    ollama_processes = list(psutil.Process(p) for p in ollama_process_ids)
    ollama_names = list( p.name() for p in ollama_processes)
    num_cores = psutil.cpu_count(logical=True)

    timestamps = []
    cpu_percentages_process = []
    cpu_percentages_total = []
    cpu_percentages_ollama = []
    memory_mb_process = []
    memory_mb_total = []
    memory_mb_ollama = []

    stop_monitoring = threading.Event()

    # Sammle Daten im Hintergrund
    def poll_metrics():
        start_time = time.time()

        while not stop_monitoring.is_set():
            current_time = time.time() - start_time

            cpu = round((process.cpu_percent(interval=None) / num_cores), 2)
            cpu_total = psutil.cpu_percent(interval=None)
            cpu_ollama = list(round((p.cpu_percent(interval=None) / num_cores), 2) for p in ollama_processes)
            mem = round((process.memory_info().rss / (1024 * 1024)), 2)
            mem_total = psutil.virtual_memory().percent
            mem_ollama = list(round((p.memory_info().rss / (1024*1024)), 2) for p in ollama_processes)

            timestamps.append(round(current_time, 2))
            cpu_percentages_process.append(cpu)
            cpu_percentages_total.append(cpu_total)
            cpu_percentages_ollama.append(cpu_ollama)
            memory_mb_process.append(mem)
            memory_mb_total.append(mem_total)
            memory_mb_ollama.append(mem_ollama)

            time.sleep(interval)

    monitor_thread = threading.Thread(target=poll_metrics, daemon=True)
    monitor_thread.start()

    # Führe die Anfrage aus
    start_execution = time.time()
    try:
        result = await call_next(request)
    finally:
        # Stelle sicher, dass Monitoring auch dann stoppt, wenn call_next crasht.
        stop_monitoring.set()
        monitor_thread.join()

    total_duration = time.time() - start_execution
    
    # Erstelle Datentabellen für Ausgabe
    cpu_metrics = {
        "time (s)": timestamps,
        "CPU Chatbot (Prozent)": cpu_percentages_process,
        "CPU total (Prozent)": cpu_percentages_total,
    }

    mem_metrics = {
        "RAM Chatbot (MB)": memory_mb_process,
        "RAM total (Prozent)": memory_mb_total
    }

    ollama_table_cpu = dt(cpu_percentages_ollama, columns=[f"CPU {proc} (Prozent)" for proc in ollama_names])
    ollama_table_mem = dt(memory_mb_ollama, columns=[f"RAM {proc} (MB)" for proc in ollama_names])
    table = ollama_table_cpu.join(dt.from_dict(cpu_metrics)).join(ollama_table_mem).join(dt.from_dict(mem_metrics)).set_index("time (s)")
    return result, table
        
    
# Dokumentiere die CPU-Last und RAM-Nutzung für eine spezifische Anfrage an den Bot
async def profiler_middleware(request: Request, call_next):
    settings = get_settings()

    # Profile die Anfrage falls 'profile=true' 
    if settings.enable_profiler and request.method == 'POST' and str(request.url)[-5:] == "/chat":
        response, metrics = await profile(request, call_next, interval=settings.profiler_sample_rate)

        # Berechne statistische Werte und gebe sie zusammen mit den gemessenen Werten als HTML-Tabellen aus
        means = metrics.mean(axis=0)
        maxs = metrics.max(axis=0)
        medians = metrics.median(axis=0)

        stats = dt(list(zip(means, medians, maxs)), columns=['Durchschnitt', 'Median', 'Maximum']).T
        stats.columns = metrics.columns

        with open(settings.profiler_output_path, 'a') as file:
            metrics.to_html(file)
            stats.to_html(file)
        return response
    
    # Andernfalls lasse es aus
    return await call_next(request)