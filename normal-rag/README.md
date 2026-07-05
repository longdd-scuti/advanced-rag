# Normal RAG FastAPI Demo

Demo nay index file `.txt` vao Qdrant collection `normal-rag`, sau do query
bang local LLM `qwen3:14b`.

Khac voi GraphRAG, normal RAG chi retrieve cac text chunks gan nhat theo vector
similarity, roi dua chunks do vao LLM de tra loi. No khong tao entity graph,
relationship graph, community report, hay global reasoning tren network quan he.

## Cai dat

```powershell
cd D:\workspaces\self_git\advanced-rag\normal-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Yeu cau local

- Ollama/OpenAI-compatible server dang chay tai `http://127.0.0.1:11434/v1`.
- Qdrant dang chay tai `http://127.0.0.1:6333`.
- Da co model:

```powershell
ollama pull nomic-embed-text
ollama pull qwen3:14b
```

Chay Qdrant bang Docker neu chua co:

```powershell
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

## Chay API

```powershell
python -m uvicorn app.main:app --reload --port 8010
```

Swagger UI:

<http://127.0.0.1:8010/docs>

## Them file input

Dat file `.txt` vao:

```text
normal-rag\input\book.txt
```

## Index

Index tat ca file `.txt` trong `input` vao collection `normal-rag`:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8010/index `
  -ContentType "application/json" `
  -Body '{"reset_collection":true}'
```

Index mot file cu the:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8010/index `
  -ContentType "application/json" `
  -Body '{"file_path":"input/book.txt","reset_collection":true}'
```

## Query

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8010/query `
  -ContentType "application/json" `
  -Body '{"question":"What is the main conflict in the story?","top_k":5,"response_type":"Concise answer"}'
```

Response co `answer`, `sources`, va `context` de debug retrieval.

## Endpoint

- `GET /health`: kiem tra API.
- `GET /status`: xem collection, models, input files.
- `POST /index`: chunk file `.txt`, embedding bang `nomic-embed-text`, upsert vao Qdrant collection `normal-rag`.
- `POST /query`: embedding question, retrieve top-k chunks tu Qdrant, goi `qwen3:14b` de tra loi.
