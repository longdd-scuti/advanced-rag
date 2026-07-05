from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid5, NAMESPACE_URL

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
COLLECTION_NAME = "normal-rag"
NO_ANSWER_MESSAGE = "I couldn't find relevant information. Please ask a different question."


def load_local_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_local_env()

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
LOCAL_LLM_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://127.0.0.1:11434/v1")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "local")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
LOCAL_QUERY_CHAT_MODEL = os.getenv("LOCAL_QUERY_CHAT_MODEL", "qwen3:14b")


app = FastAPI(
    title="Normal RAG FastAPI Demo",
    version="0.1.0",
    description="Simple file-based RAG using Qdrant and a local OpenAI-compatible LLM.",
)


class IndexRequest(BaseModel):
    file_path: str | None = Field(
        default=None,
        description="Path to a .txt file. Relative paths are resolved from the normal-rag folder.",
    )
    input_dir: str = Field(default="input", min_length=1)
    reset_collection: bool = True
    chunk_size_chars: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=2000)
    batch_size: int = Field(default=16, ge=1, le=128)
    timeout_seconds: int = Field(default=120, ge=1)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0)
    response_type: str = "Concise answer"
    timeout_seconds: int = Field(default=300, ge=1)


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def openai_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"}


def resolve_txt_files(file_path: str | None, input_dir: str) -> list[Path]:
    if file_path:
        path = Path(file_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        files = [path]
    else:
        directory = Path(input_dir)
        if not directory.is_absolute():
            directory = PROJECT_ROOT / directory
        files = sorted(directory.glob("*.txt"))

    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "Some input files do not exist.", "files": missing},
        )

    txt_files = [path for path in files if path.is_file() and path.suffix.lower() == ".txt"]
    if not txt_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No .txt files found to index.",
        )
    return txt_files


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []

    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.extend(split_long_text(current, chunk_size, overlap))
        current = paragraph
    if current:
        chunks.extend(split_long_text(current, chunk_size, overlap))
    return chunks


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_texts(texts: list[str], timeout_seconds: int) -> list[list[float]]:
    if not texts:
        return []

    url = f"{LOCAL_LLM_API_BASE.rstrip('/')}/embeddings"
    embeddings: list[list[float]] = []
    with httpx.Client(timeout=timeout_seconds, headers=openai_headers()) as client:
        response = client.post(
            url,
            json={"model": LOCAL_EMBEDDING_MODEL, "input": texts},
        )
        response.raise_for_status()
        payload = response.json()

    for item in payload.get("data", []):
        embeddings.append(item["embedding"])
    if len(embeddings) != len(texts):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Embedding server returned an unexpected number of embeddings.",
        )
    return embeddings


def ensure_collection(client: QdrantClient, vector_size: int, reset_collection: bool) -> None:
    if reset_collection and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def point_id(source_file: str, chunk_index: int, text: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{COLLECTION_NAME}:{source_file}:{chunk_index}:{text}"))


def build_prompt(question: str, context_chunks: list[dict[str, Any]], response_type: str) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[Source {index + 1}: {chunk['source_file']} chunk {chunk['chunk_index']}]\n{chunk['text']}"
        for index, chunk in enumerate(context_chunks)
    )
    system = (
        "You answer questions using only the provided context. "
        f"If the context does not contain the answer, say: {NO_ANSWER_MESSAGE} "
        "Do not invent facts. Keep the answer clear for a normal user."
    )
    user = (
        f"Desired response type: {response_type}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat_model(question: str, context_chunks: list[dict[str, Any]], response_type: str, timeout_seconds: int) -> str:
    if not context_chunks:
        return NO_ANSWER_MESSAGE

    url = f"{LOCAL_LLM_API_BASE.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout_seconds, headers=openai_headers()) as client:
        response = client.post(
            url,
            json={
                "model": LOCAL_QUERY_CHAT_MODEL,
                "messages": build_prompt(question, context_chunks, response_type),
                "temperature": 0,
                "stream": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
    answer = payload["choices"][0]["message"].get("content", "").strip()
    return answer or NO_ANSWER_MESSAGE


def search_qdrant(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int,
    score_threshold: float | None,
) -> list[Any]:
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return list(result.points)

    return list(
        client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status_endpoint() -> dict[str, Any]:
    client = qdrant_client()
    collection_exists = client.collection_exists(COLLECTION_NAME)
    points_count = 0
    if collection_exists:
        collection = client.get_collection(COLLECTION_NAME)
        points_count = collection.points_count or 0
    input_files = sorted(path.name for path in INPUT_DIR.glob("*.txt")) if INPUT_DIR.exists() else []
    return {
        "collection": COLLECTION_NAME,
        "collection_exists": collection_exists,
        "points_count": points_count,
        "qdrant_url": QDRANT_URL,
        "embedding_model": LOCAL_EMBEDDING_MODEL,
        "query_model": LOCAL_QUERY_CHAT_MODEL,
        "input_files": input_files,
    }


@app.post("/index")
def index_documents(payload: IndexRequest) -> dict[str, Any]:
    files = resolve_txt_files(payload.file_path, payload.input_dir)
    documents: list[dict[str, Any]] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        for chunk_index, chunk in enumerate(
            chunk_text(text, payload.chunk_size_chars, payload.chunk_overlap_chars)
        ):
            documents.append(
                {
                    "source_file": file.name,
                    "source_path": str(file),
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )

    if not documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input files are empty after chunking.",
        )

    client = qdrant_client()
    first_embedding = embed_texts([documents[0]["text"]], payload.timeout_seconds)[0]
    ensure_collection(client, len(first_embedding), payload.reset_collection)

    points: list[PointStruct] = [
        PointStruct(
            id=point_id(documents[0]["source_file"], documents[0]["chunk_index"], documents[0]["text"]),
            vector=first_embedding,
            payload=documents[0],
        )
    ]

    remaining = documents[1:]
    for start in range(0, len(remaining), payload.batch_size):
        batch = remaining[start : start + payload.batch_size]
        vectors = embed_texts([document["text"] for document in batch], payload.timeout_seconds)
        for document, vector in zip(batch, vectors, strict=True):
            points.append(
                PointStruct(
                    id=point_id(document["source_file"], document["chunk_index"], document["text"]),
                    vector=vector,
                    payload=document,
                )
            )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {
        "collection": COLLECTION_NAME,
        "indexed_files": [str(path) for path in files],
        "chunks_indexed": len(points),
        "embedding_model": LOCAL_EMBEDDING_MODEL,
        "reset_collection": payload.reset_collection,
    }


@app.post("/query")
def query_documents(payload: QueryRequest) -> dict[str, Any]:
    client = qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection {COLLECTION_NAME!r} does not exist. Run /index first.",
        )

    query_vector = embed_texts([payload.question], payload.timeout_seconds)[0]
    hits = search_qdrant(client, query_vector, payload.top_k, payload.score_threshold)
    context_chunks = [
        {
            "score": hit.score,
            "source_file": hit.payload.get("source_file", ""),
            "chunk_index": hit.payload.get("chunk_index", 0),
            "text": hit.payload.get("text", ""),
        }
        for hit in hits
        if hit.payload
    ]
    answer = call_chat_model(
        payload.question,
        context_chunks,
        payload.response_type,
        payload.timeout_seconds,
    )
    return {
        "answer": answer,
        "collection": COLLECTION_NAME,
        "sources": [
            {
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "score": chunk["score"],
            }
            for chunk in context_chunks
        ],
        "context": context_chunks,
    }
