from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .graphrag_cli import WORKSPACE_SOURCES_ROOT, run_graphrag


app = FastAPI(
    title="GraphRAG FastAPI Demo",
    version="0.1.0",
    description="Small FastAPI wrapper around the Microsoft GraphRAG CLI.",
)

NO_ANSWER_MESSAGE = "I couldn't find relevant information. Please ask a different question."
NO_ANSWER_PATTERNS = (
    "i don't know",
    "i do not know",
    "i am sorry but i am unable",
    "unable to answer",
    "not enough information",
    "no relevant information",
    "could not find",
    "cannot find",
    "provided data does not",
)
SOURCE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


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
    # servers, edit the source settings.yaml instead of relying on init defaults.
    source: str = Field(..., min_length=1)
    model: str = Field(default="gpt-4.1", min_length=1)
    embedding: str = Field(default="text-embedding-3-large", min_length=1)
    force: bool = False
    timeout_seconds: int | None = Field(default=300, ge=1)


class TextDocumentRequest(BaseModel):
    source: str = Field(..., min_length=1)
    filename: str = Field(default="demo.txt", min_length=1)
    text: str = Field(..., min_length=1)
    overwrite: bool = False


class IndexRequest(BaseModel):
    # `fast` is a practical default for local machines because it reduces LLM
    # usage during indexing. Use `standard` only after validating small samples.
    source: str = Field(..., min_length=1)
    method: IndexMethod = IndexMethod.fast
    verbose: bool = False
    dry_run: bool = False
    cache: bool = True
    skip_validation: bool = False
    timeout_seconds: int | None = Field(default=1800, ge=1)


class QueryRequest(BaseModel):
    source: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    method: QueryMethod = QueryMethod.global_
    response_type: str = "Multiple Paragraphs"
    community_level: int = Field(default=2, ge=0)
    data: str | None = None
    verbose: bool = False
    timeout_seconds: int | None = Field(default=300, ge=1)


def source_workspace_root(source: str) -> Path:
    """Return the workspace root for a required source name."""

    normalized = source.strip()
    if not SOURCE_NAME_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="source may only contain letters, numbers, hyphens, and underscores",
        )
    return WORKSPACE_SOURCES_ROOT / normalized


def source_input_dir(source: str) -> Path:
    return source_workspace_root(source) / "input"


def available_sources() -> list[str]:
    if not WORKSPACE_SOURCES_ROOT.exists():
        return []
    return sorted(
        path.name for path in WORKSPACE_SOURCES_ROOT.iterdir() if path.is_dir()
    )


def safe_input_path(filename: str, input_dir: Path) -> Path:
    """Normalize uploaded document names so writes stay inside the source input."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="filename must contain at least one letter or number",
        )
    if not normalized.lower().endswith(".txt"):
        normalized = f"{normalized}.txt"
    return input_dir / normalized


def latest_output_dir(workspace_root: Path) -> str | None:
    """Return the newest GraphRAG output folder, if indexing has run."""

    output_root = workspace_root / "output"
    if not output_root.exists():
        return None

    if any(output_root.glob("*.parquet")):
        return str(output_root)

    candidates = [path for path in output_root.iterdir() if path.is_dir()]
    if not candidates:
        return str(output_root)
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def user_facing_answer(stdout: str) -> str:
    """Convert GraphRAG CLI output into a simple answer for API consumers."""

    answer = stdout.strip()
    if not answer:
        return NO_ANSWER_MESSAGE

    lowered = answer.lower()
    if any(pattern in lowered for pattern in NO_ANSWER_PATTERNS):
        return NO_ANSWER_MESSAGE

    answer = re.sub(r"\s*\[Data:[^\]]+\]", "", answer)
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    return answer.strip() or NO_ANSWER_MESSAGE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/workspace/status")
def workspace_status(source: str) -> dict[str, object]:
    # This endpoint is intentionally cheap: it only checks local files and does
    # not call GraphRAG or any model server.
    workspace_root = source_workspace_root(source)
    input_dir = workspace_root / "input"
    input_files = (
        sorted(path.name for path in input_dir.glob("*.txt")) if input_dir.exists() else []
    )
    return {
        "source": source,
        "available_sources": available_sources(),
        "workspace": str(workspace_root),
        "settings_yaml": (workspace_root / "settings.yaml").exists(),
        "env_file": (workspace_root / ".env").exists(),
        "input_files": input_files,
        "latest_output": latest_output_dir(workspace_root),
    }


@app.post("/workspace/init")
def init_workspace(payload: InitRequest) -> dict[str, object]:
    # This creates settings.yaml, .env, input and prompt templates for a source.
    # It does not index data or call an LLM.
    workspace_root = source_workspace_root(payload.source)
    args = [
        "init",
        "--root",
        str(workspace_root),
        "--model",
        payload.model,
        "--embedding",
        payload.embedding,
    ]
    if payload.force:
        args.append("--force")
    return run_graphrag(args, workspace_root, payload.timeout_seconds)


@app.post("/documents/text")
def add_text_document(payload: TextDocumentRequest) -> dict[str, str]:
    # GraphRAG reads .txt files from the source input when `input.type: text`.
    input_dir = source_input_dir(payload.source)
    input_dir.mkdir(parents=True, exist_ok=True)
    target = safe_input_path(payload.filename, input_dir)
    if target.exists() and not payload.overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{target.name} already exists. Set overwrite=true to replace it.",
        )
    target.write_text(payload.text, encoding="utf-8")
    return {
        "source": payload.source or "",
        "filename": target.name,
        "path": str(target),
    }


@app.post("/documents/sample")
def add_sample_document(
    overwrite: bool = False,
    source: str = "",
) -> dict[str, str]:
    if not source:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="source is required",
        )
    sample = """Contoso Analytics builds internal tools for retail operations teams.

The Atlas product helps store managers monitor inventory risk, late supplier shipments,
and regional sales anomalies. Atlas integrates with warehouse systems, point-of-sale
events, and employee task queues.

Mira Patel leads the Atlas data platform. Jonah Reed owns supplier integrations.
The operations team uses GraphRAG to connect incidents, stores, suppliers, and product
categories so analysts can ask broader questions about recurring business issues.
"""
    return add_text_document(
        TextDocumentRequest(
            source=source,
            filename="contoso_atlas.txt",
            text=sample,
            overwrite=overwrite,
        )
    )


@app.post("/index")
def index_workspace(payload: IndexRequest) -> dict[str, object]:
    # This is the expensive step. With local models, keep the sample small first
    # and increase timeout_seconds because consumer GPUs can be much slower than
    # hosted APIs for many sequential extraction calls.
    workspace_root = source_workspace_root(payload.source)
    args = ["index", "--root", str(workspace_root), "--method", payload.method.value]
    if payload.verbose:
        args.append("--verbose")
    if payload.dry_run:
        args.append("--dry-run")
    args.append("--cache" if payload.cache else "--no-cache")
    if payload.skip_validation:
        args.append("--skip-validation")
    return run_graphrag(args, workspace_root, payload.timeout_seconds)


@app.post("/query")
def query_workspace(payload: QueryRequest) -> dict[str, object]:
    # Query requires a completed index in source output. `local` is often the
    # better first choice for factual questions; `global` summarizes themes.
    workspace_root = source_workspace_root(payload.source)
    args = [
        "query",
        payload.question,
        "--root",
        str(workspace_root),
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
    result = run_graphrag(args, workspace_root, payload.timeout_seconds)
    return {
        "source": payload.source,
        "answer": user_facing_answer(str(result["stdout"])),
        **result,
    }
