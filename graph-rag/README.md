# GraphRAG FastAPI Demo

Demo nay boc Microsoft GraphRAG CLI bang FastAPI. Luong chay theo tai lieu chinh thuc:

1. Cai `graphrag`.
2. Tao workspace rieng cho tung source tai `workspace_sources/<source>`.
3. Dat file `.txt` vao `workspace_sources/<source>/input`.
4. Chay `graphrag index` rieng cho tung source.
5. Query bang `source` tuong ung.

Tai lieu tham khao: <https://microsoft.github.io/graphrag/get_started/>

## Cai dat

Tren Windows, tao virtual environment truc tiep trong project. Thu muc `.venv`
da nam trong `.gitignore`, nen khong bi commit len git.

```cmd
cd D:\workspaces\self_git\advanced-rag\graph-rag
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Neu chay bang PowerShell va lenh activate tren khong duoc nhan, dung:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Khoi tao GraphRAG workspace

API bat buoc truyen `source` khi thao tac voi workspace. Moi source la mot
workspace con trong `workspace_sources`.

Repo da chuan bi san 2 source mau:

```text
workspace_sources\book\input\book.txt
workspace_sources\contoso_atlas\input\contoso_atlas.txt
```

Neu muon tao source moi, vi du `my_source`:

```powershell
New-Item -ItemType Directory -Force .\workspace_sources\my_source\input
Copy-Item .\templates\settings.yaml .\workspace_sources\my_source\settings.yaml
Copy-Item .\templates\.env.example .\workspace_sources\my_source\.env
Copy-Item .\my_source.txt .\workspace_sources\my_source\input\my_source.txt
```

Moi source can co `.env` rieng. Voi Ollama local, cac bien chinh la:

```dotenv
LOCAL_LLM_API_BASE=http://127.0.0.1:11434/v1
LOCAL_EMBEDDING_API_BASE=http://127.0.0.1:11434/v1
LOCAL_LLM_API_KEY=local
LOCAL_INDEX_CHAT_MODEL=qwen3:8b
LOCAL_QUERY_CHAT_MODEL=qwen3:14b
LOCAL_EMBEDDING_MODEL=nomic-embed-text
```

## Index rieng tung source

Chay index cho tung source:

```powershell
graphrag index --root .\workspace_sources\book --method fast --cache
graphrag index --root .\workspace_sources\contoso_atlas --method fast --cache
```

Hoac qua API:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/index `
  -ContentType "application/json" `
  -Body '{"source":"book","method":"fast","verbose":true,"cache":true,"timeout_seconds":3600}'
```

## Chay FastAPI

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

Mo Swagger UI:

<http://127.0.0.1:8000/docs>

## Chay local tren may cua ban

May i7 14700, RTX 4060 Ti 16GB, RAM 64GB du de demo GraphRAG local voi bo tai
lieu nho/vua. Diem can luu y la GraphRAG indexing goi model nhieu lan de trich
xuat entity/relationship va tao community report, nen latency local se cao hon
API cloud.

Huong nen di:

- Dung mot OpenAI-compatible local server nhu Ollama, LM Studio hoac vLLM.
- Dung `qwen3:8b` cho indexing de nhanh hon, va `qwen3:14b` cho query de
  cau tra loi tot hon.
- Embedding nen dung model embedding rieng, vi chat model khong phai luc nao cung
  co endpoint embedding tot.
- Bat dau voi `method: fast`, input nho, roi moi tang kich thuoc du lieu.
- Model local phai tra JSON/structured output on dinh. Neu output hay sai format,
  GraphRAG index se fail du GPU van con du VRAM.

File mau cho local OpenAI-compatible server:

```text
templates\settings.local-openai-compatible.example.yaml
```

Them cac bien nay vao `.env` cua source, vi du
`workspace_sources\book\.env`. Template config nam o `templates\settings.yaml`.

```dotenv
LOCAL_LLM_API_BASE=http://127.0.0.1:11434/v1
LOCAL_EMBEDDING_API_BASE=http://127.0.0.1:11434/v1
LOCAL_LLM_API_KEY=local
LOCAL_INDEX_CHAT_MODEL=qwen3:8b
LOCAL_QUERY_CHAT_MODEL=qwen3:14b
LOCAL_EMBEDDING_MODEL=nomic-embed-text
```

## Test nhanh bang API

Kiem tra server:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Kiem tra source:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/workspace/status?source=book"
Invoke-RestMethod "http://127.0.0.1:8000/workspace/status?source=contoso_atlas"
```

Them document vao mot source qua API:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/documents/text `
  -ContentType "application/json" `
  -Body '{"source":"book","filename":"book.txt","text":"Your text here","overwrite":true}'
```

Query mot source cu the:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"source":"contoso_atlas","question":"Who owns supplier integrations?","method":"local","response_type":"Single Sentence"}'
```

Response se co truong `answer` de hien thi cho nguoi dung. Truong `stdout` va
`stderr` van duoc giu lai de debug GraphRAG CLI.

```json
{
  "answer": "Jonah Reed owns supplier integrations.",
  "stdout": "Jonah Reed owns supplier integrations [Data: Sources (0)].\n"
}
```

## Endpoint chinh

- `GET /health`: kiem tra server.
- `GET /workspace/status?source=book`: xem workspace, input files, output moi nhat.
- `POST /workspace/init`: goi `graphrag init`, bat buoc truyen `source`.
- `POST /documents/sample?source=contoso_atlas`: tao file sample cho source.
- `POST /documents/text`: them file `.txt` tuy y vao source.
- `POST /index`: goi `graphrag index`, bat buoc truyen `source`.
- `POST /query`: goi `graphrag query`, bat buoc truyen `source`.

## Luu y

GraphRAG indexing co the dung nhieu token. Nen bat dau bang sample nho va method
`fast` truoc khi index bo tai lieu lon.
