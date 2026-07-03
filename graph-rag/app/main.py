from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .graphrag_cli import INPUT_DIR, WORKSPACE_ROOT, run_graphrag


app = FastAPI(
    title="GraphRAG FastAPI Demo",
    version="0.1.0",
    description="Small FastAPI wrapper around the Microsoft GraphRAG CLI.",
)


# These values map directly to `graphrag index --method ...`.
class IndexMethod(str, Enum):
    standard = "standard"
    fast = "fast"
    standard_update = "standard-update"
    fast_update = "fast-update"


# These values map directly to `graphrag query --method ...`.
class QueryMethod(str, Enum):
    local = "local"
    global_ = "global"
    drift = "drift"
    basic = "basic"


class InitRequest(BaseModel):
    # Defaults follow the official quickstart. For local OpenAI-compatible
    # servers, edit workspace/settings.yaml instead of relying on init defaults.
    model: str = Field(default="gpt-4.1", min_length=1)
    embedding: str = Field(default="text-embedding-3-large", min_length=1)
    force: bool = False
    timeout_seconds: int | None = Field(default=300, ge=1)


class TextDocumentRequest(BaseModel):
    filename: str = Field(default="demo.txt", min_length=1)
    text: str = Field(..., min_length=1)
    overwrite: bool = False


class IndexRequest(BaseModel):
    # `fast` is a practical default for local machines because it reduces LLM
    # usage during indexing. Use `standard` only after validating small samples.
    method: IndexMethod = IndexMethod.fast
    verbose: bool = False
    dry_run: bool = False
    cache: bool = True
    skip_validation: bool = False
    timeout_seconds: int | None = Field(default=1800, ge=1)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    method: QueryMethod = QueryMethod.global_
    response_type: str = "Multiple Paragraphs"
    community_level: int = Field(default=2, ge=0)
    data: str | None = None
    verbose: bool = False
    timeout_seconds: int | None = Field(default=300, ge=1)


def safe_input_path(filename: str) -> Path:
    """Normalize uploaded document names so writes stay inside workspace/input."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="filename must contain at least one letter or number",
        )
    if not normalized.lower().endswith(".txt"):
        normalized = f"{normalized}.txt"
    return INPUT_DIR / normalized


def latest_output_dir() -> str | None:
    """Return the newest GraphRAG output folder, if indexing has run."""

    output_root = WORKSPACE_ROOT / "output"
    if not output_root.exists():
        return None

    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        return str(output_root)
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workspace/status")
def workspace_status() -> dict[str, object]:
    # This endpoint is intentionally cheap: it only checks local files and does
    # not call GraphRAG or any model server.
    input_files = sorted(path.name for path in INPUT_DIR.glob("*.txt")) if INPUT_DIR.exists() else []
    return {
        "workspace": str(WORKSPACE_ROOT),
        "settings_yaml": (WORKSPACE_ROOT / "settings.yaml").exists(),
        "env_file": (WORKSPACE_ROOT / ".env").exists(),
        "input_files": input_files,
        "latest_output": latest_output_dir(),
    }


@app.post("/workspace/init")
def init_workspace(payload: InitRequest) -> dict[str, object]:
    # This creates workspace/settings.yaml, workspace/.env, workspace/input and
    # prompt templates. It does not index data or call an LLM.
    args = [
        "init",
        "--root",
        str(WORKSPACE_ROOT),
        "--model",
        payload.model,
        "--embedding",
        payload.embedding,
    ]
    if payload.force:
        args.append("--force")
    return run_graphrag(args, payload.timeout_seconds)


@app.post("/documents/text")
def add_text_document(payload: TextDocumentRequest) -> dict[str, str]:
    # GraphRAG reads .txt files from workspace/input when `input.type: text`.
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = safe_input_path(payload.filename)
    if target.exists() and not payload.overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{target.name} already exists. Set overwrite=true to replace it.",
        )
    target.write_text(payload.text, encoding="utf-8")
    return {"filename": target.name, "path": str(target)}


@app.post("/documents/sample")
def add_sample_document(overwrite: bool = False) -> dict[str, str]:
    sample = """Contoso Analytics builds internal tools for retail operations teams.

The Atlas product helps store managers monitor inventory risk, late supplier shipments,
and regional sales anomalies. Atlas integrates with warehouse systems, point-of-sale
events, and employee task queues.

Mira Patel leads the Atlas data platform. Jonah Reed owns supplier integrations.
The operations team uses GraphRAG to connect incidents, stores, suppliers, and product
categories so analysts can ask broader questions about recurring business issues.
"""
    return add_text_document(
        TextDocumentRequest(filename="contoso_atlas.txt", text=sample, overwrite=overwrite)
    )


@app.post("/index")
def index_workspace(payload: IndexRequest) -> dict[str, object]:
    # This is the expensive step. With local models, keep the sample small first
    # and increase timeout_seconds because consumer GPUs can be much slower than
    # hosted APIs for many sequential extraction calls.
    args = ["index", "--root", str(WORKSPACE_ROOT), "--method", payload.method.value]
    if payload.verbose:
        args.append("--verbose")
    if payload.dry_run:
        args.append("--dry-run")
    args.append("--cache" if payload.cache else "--no-cache")
    if payload.skip_validation:
        args.append("--skip-validation")
    return run_graphrag(args, payload.timeout_seconds)


@app.post("/query")
def query_workspace(payload: QueryRequest) -> dict[str, object]:
    # Query requires a completed index in workspace/output. `local` is often the
    # better first choice for factual questions; `global` summarizes themes.
    args = [
        "query",
        payload.question,
        "--root",
        str(WORKSPACE_ROOT),
        "--method",
        payload.method.value,
        "--response-type",
        payload.response_type,
        "--community-level",
        str(payload.community_level),
    ]
    if payload.data:
        args.extend(["--data", payload.data])
    if payload.verbose:
        args.append("--verbose")
    return run_graphrag(args, payload.timeout_seconds)
