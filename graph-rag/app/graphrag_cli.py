from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import HTTPException, status


# All GraphRAG CLI commands are run against this local workspace. Keeping the
# GraphRAG root separate from the API code makes it easy to reset/re-index data.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_SOURCES_ROOT = PROJECT_ROOT / "workspace_sources"


def find_graphrag_cli() -> str:
    """Find the GraphRAG executable used by init/index/query endpoints."""

    # Useful when the API runs from one Python environment but GraphRAG is
    # installed in another, e.g. C:\venvs\advanced-rag-graphrag on Windows.
    configured = os.getenv("GRAPHRAG_CLI")
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return str(configured_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GRAPHRAG_CLI is set but does not exist: {configured}",
        )

    # Prefer the executable next to the Python interpreter that started uvicorn.
    # This avoids accidentally using a different global GraphRAG installation.
    executable_name = "graphrag.exe" if os.name == "nt" else "graphrag"
    next_to_python = Path(sys.executable).parent / executable_name
    if next_to_python.exists():
        return str(next_to_python)

    local_venv = (
        PROJECT_ROOT
        / ".venv"
        / ("Scripts" if os.name == "nt" else "bin")
        / executable_name
    )
    if local_venv.exists():
        return str(local_venv)

    discovered = shutil.which("graphrag")
    if discovered:
        return discovered

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=(
            "GraphRAG CLI was not found. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ),
    )


def run_graphrag(
    args: list[str],
    workspace_root: Path,
    timeout_seconds: int | None = None,
) -> dict[str, str | int]:
    """Run a GraphRAG CLI command and return stdout/stderr for API debugging."""

    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "input").mkdir(parents=True, exist_ok=True)

    # Keep command construction as a list. This avoids shell quoting issues and
    # prevents user-provided query text from being interpreted as shell syntax.
    command = [find_graphrag_cli(), *args]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "message": f"GraphRAG command timed out after {timeout_seconds} seconds.",
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
            },
        ) from exc

    result: dict[str, str | int] = {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

    if completed.returncode != 0:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=result)

    return result
