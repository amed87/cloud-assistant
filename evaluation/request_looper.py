"""Lasten-/Performance-Loop für den lokalen Chatbot-Server.

Läuft gegen den mit uvicorn gestarteten API-Server und schickt in einer
vordefinierten Matrix wiederholte Requests an verschiedene Endpunkte.

Aufruf: python evaluation/request_looper.py

Wichtige Anpassungen im Skript:
- thread_range: Thread-Anzahl pro Testlauf
- endpoints: zu prüfende API-Endpunkte
- messages: Testfragen
- repeats: Wiederholungen je Anfrage
- profiler_output_path / profiler_output_format: Ausgabe des Profilers
"""

import requests
import httpx
import os
import asyncio
from pathlib import Path
import psutil
from app.config.settings import get_settings

PROJECT_ROOT= Path(__file__).resolve().parents[1]
VENV = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
base_url = "http://127.0.0.1:8000"
endpoints = {
             "chat": f"{base_url}/chat",
             "quick": f"{base_url}/chat/quick",
             "stream": f"{base_url}/stream"
             }
message_options = {
    "unknown_simple": "Hello",
    "unknown_harder": "Where would I find the company manual?",
    "known_simple": "kubernetes",
    "known_harder": "What is Kubernetes for?"
}

# Die Laufzeit-Umgebung wird vor dem Start des Servers gezielt überschrieben,
# damit die Performance-Tests mit der gewünschten Threadzahl und Profil-Ausgabe laufen.
def setup_env(num_threads: int, profiler_output_path: str, profiler_output_format: str):
    os.environ["ollama_num_threads"] = str(num_threads)
    os.environ["PROFILER_OUTPUT_PATH"] = profiler_output_path
    os.environ["PROFILER_OUTPUT_FORMAT"] = profiler_output_format
    return True

# Der Server wird nur neu gestartet, wenn sich die für den Test relevanten
# Konfigurationen geändert haben. Dadurch bleiben wiederholte Anfragen schnell.
async def setup(num_threads: int, profiler_output_path: str, profiler_output_format: str, server: asyncio.subprocess.Process = None) -> asyncio.subprocess.Process:
    needs_restart = (
        server is None
        or not os.environ["ollama_num_threads"] == str(num_threads) 
        or not os.environ["PROFILER_OUTPUT_PATH"] == profiler_output_path 
        or not os.environ["PROFILER_OUTPUT_FORMAT"] == profiler_output_format
    )

    if needs_restart:
        if server: server.terminate()
        kill_port()
        setup_env(num_threads, profiler_output_path, profiler_output_format)
        server = await asyncio.create_subprocess_exec(str(VENV), "-m", "uvicorn", "app.main:app")
    return server

async def wait_for_ready(client: httpx.AsyncClient, retries=30, delay=1):
    # Der Server braucht kurz Zeit zum Hochfahren; mit wenigen Retries wird
    # ein kurzer Start-Delay robust abgefangen, ohne den Test zu blockieren.
    for _ in range(retries):
        try:
            await client.get(f"{base_url}/docs")  
            return  # Server antwortet → ready
        except httpx.ConnectError:
            await asyncio.sleep(delay)
    raise RuntimeError("Server wurde nicht bereit")

def kill_port(port=8000):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.status == 'LISTEN':
            p = psutil.Process(conn.pid)
            p.kill()

# Die Testmatrix besteht aus allen Kombinationen aus Threadzahl, Endpoint,
# Nachricht und Conversation-ID. Jede Wiederholung erhält dabei einen eigenen Eintrag.
def build_requests(thread_range: list[int], endpoints: list[str], messages: list[str], conversation_ids: list[int], repeats: int) -> list[dict]:
    reqs = []
    for num in thread_range:
        for endpoint in endpoints:
            for message in messages:
                for id in conversation_ids:
                    for rep in range(1, repeats+1):
                        req = {
                            "threads": num,
                            "endpoint": endpoint,
                            "conversation_id": str(id),
                            "message": message,
                            "repeat": rep,
                            "response": None
                        }
                        reqs.append(req)
    return reqs

async def send_requests(url: str, data: dict):
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        response = await client.post(url, json=data)
        return response

# Jede eingetragene Anfrage wird nacheinander abgearbeitet. Für die erste
# Wiederholung einer Konfiguration wird der Server neu vorbereitet; danach werden nur noch
# Requests gesendet und das Ergebnis in der Matrix gespeichert.
async def loop_requests(thread_range: list[int], endpoints: list[str], messages: list[str], conversation_ids: list[int], repeats: int, profiler_output_path: str, profiler_output_format: str) -> list[dict]:
    reqs = build_requests(thread_range, endpoints, messages, conversation_ids, repeats)
    server = None

    for req in reqs:
        try:
            if req["repeat"] == 1:  # Nur bei der ersten Wiederholung der Anfrage wird die Server-Konfiguration neu gesetzt.
                server = await setup(req["threads"], profiler_output_path, profiler_output_format)
            print("setup done")
            async with httpx.AsyncClient(base_url=base_url) as client: 
                await wait_for_ready(client)
            req["response"] = await send_requests(req["endpoint"], {"conversation_id": req["conversation_id"], "message": req["message"]})
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            print("Error:", e)

    if server: server.terminate()
    kill_port()
    return reqs


async def main():
    # Checke wie die Anzahl Threads Load und Geschwindigkeit beeinflusst
    reqs = await loop_requests(range(4,5), [endpoints["quick"], endpoints["chat"]], [message_options["known_simple"]], [0], 50, str(PROJECT_ROOT/".."/"eval"/"num_threads_5.html"), "html")

if __name__=='__main__':
    asyncio.run(main())

