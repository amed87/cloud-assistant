"""Load and performance loop for the local chatbot server.

Run against the API server started with uvicorn and send repeated requests
to different endpoints using a predefined matrix.

Run with: python evaluation/request_looper.py

Important script settings:
- thread_range: Number of threads per test run
- endpoints: API endpoints to test
- messages: Test questions
- repeats: Repetitions per request
- profiler_output_path / profiler_output_format: Profiler output settings
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

# Override the runtime environment before starting the server so performance
# tests use the requested thread count and profiler output settings.
def setup_env(num_threads: int, profiler_output_path: str, profiler_output_format: str):
    os.environ["ollama_num_threads"] = str(num_threads)
    os.environ["PROFILER_OUTPUT_PATH"] = profiler_output_path
    os.environ["PROFILER_OUTPUT_FORMAT"] = profiler_output_format
    return True

# Restart the server only when test-relevant settings change, keeping repeated
# requests fast.
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
    # Allow a short startup delay and retry a few times without blocking the test.
    for _ in range(retries):
        try:
            await client.get(f"{base_url}/docs")  
            return  # The server responds and is ready.
        except httpx.ConnectError:
            await asyncio.sleep(delay)
    raise RuntimeError("Server did not become ready.")

def kill_port(port=8000):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.status == 'LISTEN':
            p = psutil.Process(conn.pid)
            p.kill()

# The test matrix contains every combination of thread count, endpoint, message,
# and conversation ID. Each repetition gets its own entry.
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

# Process requests sequentially. Prepare the server for the first repetition of
# each configuration, then send requests and store each result in the matrix.
async def loop_requests(thread_range: list[int], endpoints: list[str], messages: list[str], conversation_ids: list[int], repeats: int, profiler_output_path: str, profiler_output_format: str) -> list[dict]:
    reqs = build_requests(thread_range, endpoints, messages, conversation_ids, repeats)
    server = None

    for req in reqs:
        try:
            if req["repeat"] == 1:  # Apply the server configuration on the first repetition.
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
    # Check how thread count affects load and response speed.
    reqs = await loop_requests(range(4,5), [endpoints["quick"], endpoints["chat"]], [message_options["known_simple"]], [0], 50, str(PROJECT_ROOT/".."/"eval"/"num_threads_5.html"), "html")

if __name__=='__main__':
    asyncio.run(main())

