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

import csv
import httpx
import os
import asyncio
import signal
import subprocess
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DATA = PROJECT_ROOT.parent / "eval" / "data"
EVAL_OUTPUT = PROJECT_ROOT.parent / "eval" / "output"
WORKLOAD_FILE = EVAL_DATA / "synthetic_load_workload.csv"
VENV = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
base_url = "http://127.0.0.1:8000"

endpoints = {
             "chat": f"{base_url}/chat",
             "quick": f"{base_url}/chat/quick",
             "stream": f"{base_url}/stream"
             }

def load_workload_cases(path: Path = WORKLOAD_FILE) -> list[dict[str, str]]:
    """Load endpoint-independent messages and labels from the workload CSV."""
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"case_id", "category", "message"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Workload CSV is missing required columns: {missing}.")
        return list(reader)

message_options = {
    "unknown_simple": "Hello",
    "unknown_harder": "Where would I find the company manual?",
    "known_simple": "kubernetes",
    "known_harder": "What is Kubernetes for?"
}

# Override the runtime environment before starting the server so performance
# tests use the requested thread count and profiler output settings.
def setup_env(
    num_threads: int,
    profiler_output_path: str,
    profiler_output_format: str,
    run_id: str,
) -> None:
    os.environ["ollama_num_threads"] = str(num_threads)
    os.environ["PROFILER_OUTPUT_PATH"] = profiler_output_path
    os.environ["PROFILER_OUTPUT_FORMAT"] = profiler_output_format
    os.environ["PROFILER_RUN_ID"] = run_id
    os.environ["ENABLE_PROFILER"] = "true"

# Restart the server only when test-relevant settings change, keeping repeated
# requests fast.
async def setup(
    num_threads: int,
    profiler_output_path: str,
    profiler_output_format: str,
    server: asyncio.subprocess.Process | None = None,
    run_id: str | None = None,
) -> asyncio.subprocess.Process:
    run_id = run_id or uuid.uuid4().hex
    needs_restart = (
        server is None
        or os.environ.get("ollama_num_threads") != str(num_threads)
        or os.environ.get("PROFILER_OUTPUT_PATH") != profiler_output_path
        or os.environ.get("PROFILER_OUTPUT_FORMAT") != profiler_output_format
        or os.environ.get("PROFILER_RUN_ID") != run_id
        or os.environ.get("ENABLE_PROFILER", "").lower() != "true"
    )

    if needs_restart:
        if server is not None:
            await stop_server(server)
        setup_env(
            num_threads,
            profiler_output_path,
            profiler_output_format,
            run_id,
        )
        process_options = {}
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        server = await asyncio.create_subprocess_exec(
            str(VENV),
            "-m",
            "uvicorn",
            "app.main:app",
            **process_options,
        )
    return server

async def wait_for_ready(
    client: httpx.AsyncClient,
    server: asyncio.subprocess.Process,
    retries: int = 30,
    delay: float = 1.0,
) -> None:
    # Allow a short startup delay and retry a few times without blocking the test.
    for _ in range(retries):
        if server.returncode is not None:
            raise RuntimeError(
                f"Server exited during startup with code {server.returncode}."
            )
        try:
            response = await client.get(f"{base_url}/docs")
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        await asyncio.sleep(delay)
    raise RuntimeError("Server did not become ready.")


async def stop_server(
    server: asyncio.subprocess.Process,
    timeout: float = 15.0,
) -> None:
    if server.returncode is not None:
        return

    stop_signal = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
    try:
        server.send_signal(stop_signal)
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(server.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        server.kill()
        await server.wait()

# The test matrix contains every combination of thread count, endpoint, message,
# and conversation ID. Each repetition gets its own entry.
def build_requests(
    thread_range: list[int] | range,
    endpoints: list[str],
    messages: list[str | dict[str, str]],
    conversation_ids: list[int | str],
    repeats: int,
) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be at least one")

    reqs = []
    for num in thread_range:
        for endpoint in endpoints:
            for message_case in messages:
                if isinstance(message_case, str):
                    message = message_case
                    case_id = None
                    category = None
                    expected_entry_id = None
                    expected_answerable = None
                else:
                    message = message_case["message"]
                    case_id = message_case.get("case_id")
                    category = message_case.get("category")
                    expected_entry_id = message_case.get("expected_entry_id") or None
                    expected_answerable = message_case.get("expected_answerable")

                for conversation_id in conversation_ids:
                    for rep in range(1, repeats+1):
                        req = {
                            "threads": num,
                            "endpoint": endpoint,
                            "conversation_id": str(conversation_id),
                            "message": message,
                            "case_id": case_id,
                            "category": category,
                            "expected_entry_id": expected_entry_id,
                            "expected_answerable": expected_answerable,
                            "repeat": rep,
                            "response": None
                        }
                        reqs.append(req)
    return reqs

async def send_requests(
    url: str,
    data: dict,
    run_id: str | None = None,
    case_id: str | None = None,
):
    headers = {}
    if run_id:
        headers["X-Run-ID"] = run_id
    if case_id:
        headers["X-Case-ID"] = case_id
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        response = await client.post(url, json=data, headers=headers)
        return response

# Process requests. Prepare the server for the first repetition of
# each configuration, then send requests and store each result in the matrix.
async def loop_requests(
    thread_range: list[int] | range,
    endpoints: list[str],
    messages: list[str | dict[str, str]],
    conversation_ids: list[int | str],
    repeats: int,
    profiler_output_path: str,
    profiler_output_format: str,
) -> list[dict]:
    reqs = build_requests(thread_range, endpoints, messages, conversation_ids, repeats)
    server = None
    scenario_id = uuid.uuid4().hex
    thread_values = list(dict.fromkeys(thread_range))

    for thread_index, num_threads in enumerate(thread_values):
        thread_requests = [
            request for request in reqs
            if request["threads"] == num_threads
        ]
        if not thread_requests:
            continue

        run_id = (
            scenario_id
            if len(thread_values) == 1
            else f"{scenario_id}-threads-{num_threads}-{thread_index + 1}"
        )
        output_path = Path(profiler_output_path)
        run_output_path = str(
            output_path.with_name(
                f"{output_path.stem}-{run_id}{output_path.suffix}"
            )
        )
        try:
            server = await setup(
                num_threads,
                run_output_path,
                profiler_output_format,
                server,
                run_id,
            )
            async with httpx.AsyncClient(base_url=base_url) as client:
                await wait_for_ready(client, server)
                for request in thread_requests:
                    request["run_id"] = run_id
                    try:
                        response = await send_requests(
                            str(request["endpoint"]),
                            {
                                "conversation_id": request["conversation_id"],
                                "message": request["message"],
                            },
                            run_id=run_id,
                            case_id=(
                                str(request["case_id"])
                                if request["case_id"]
                                else None
                            ),
                        )
                        request["response"] = response
                        request["status_code"] = response.status_code
                    except (httpx.RequestError, httpx.HTTPStatusError) as ex:
                        request["request_error"] = str(ex)
        finally:
            if server is not None:
                await stop_server(server)
                server = None
    return reqs


async def main() -> None:
    messages = load_workload_cases()
    selected_endpoints = [endpoints["quick"]]
    thread_range = range(4, 5)
    conversation_ids = [0]
    repeats = 10

    for endpoint in selected_endpoints:
        endpoint_name = next(
            name for name, url in endpoints.items() if url == endpoint
        )
        await loop_requests(
            thread_range=thread_range,
            endpoints=[endpoint],
            messages=messages,
            conversation_ids=conversation_ids,
            repeats=repeats,
            profiler_output_path=str(EVAL_OUTPUT / f"load-test-{endpoint_name}.csv"),
            profiler_output_format="csv",
        )

if __name__ == "__main__":
    asyncio.run(main())
