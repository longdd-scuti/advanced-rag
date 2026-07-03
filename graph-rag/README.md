# GraphRAG FastAPI Demo

Demo nay boc Microsoft GraphRAG CLI bang FastAPI. Luong chay theo tai lieu chinh thuc:

1. Cai `graphrag`.
2. Chay `graphrag init`.
3. Dat file `.txt` vao `workspace/input`.
4. Chay `graphrag index`.
5. Query bang `graphrag query`.

Tai lieu tham khao: <https://microsoft.github.io/graphrag/get_started/>

## Cai dat

Tren Windows, nen dat venv o path ngan vi dependency cua GraphRAG co mot so file
path rat sau. Cai trong `.venv` nam ben duoi project dai co the cham loi
`No such file or directory` khi pip cai `litellm`.

```powershell
cd C:\Users\Admin\DATA\Projects\my_seft\advanced-rag\graph-rag
python -m venv C:\venvs\advanced-rag-graphrag
C:\venvs\advanced-rag-graphrag\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Khoi tao GraphRAG workspace

Workspace demo da duoc init trong `workspace` voi `gpt-4.1-mini` va
`text-embedding-3-small`. Neu muon tao lai config:

```powershell
graphrag init --root .\workspace --model gpt-4.1-mini --embedding text-embedding-3-small --force
```

Sau khi init, sua `workspace\.env`:

```dotenv
GRAPHRAG_API_KEY=<YOUR_OPENAI_OR_AZURE_OPENAI_KEY>
```

Neu muon dung model re hon cho demo, co the init qua API voi `gpt-4.1-mini` va
`text-embedding-3-small`.

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
- Bat dau voi model instruct 7B/8B hoac 14B quantized, temperature 0.
- Embedding nen dung model embedding rieng, vi chat model khong phai luc nao cung
  co endpoint embedding tot.
- Bat dau voi `method: fast`, input nho, roi moi tang kich thuoc du lieu.
- Model local phai tra JSON/structured output on dinh. Neu output hay sai format,
  GraphRAG index se fail du GPU van con du VRAM.

File mau cho local OpenAI-compatible server:

```text
workspace\settings.local-openai-compatible.example.yaml
```

Them cac bien nay vao `workspace\.env`, roi copy 2 block `completion_models` va
`embedding_models` tu file mau vao `workspace\settings.yaml`:

```dotenv
LOCAL_LLM_API_BASE=http://127.0.0.1:11434/v1
LOCAL_EMBEDDING_API_BASE=http://127.0.0.1:11434/v1
LOCAL_LLM_API_KEY=local
LOCAL_CHAT_MODEL=qwen2.5:7b
LOCAL_EMBEDDING_MODEL=nomic-embed-text
```

## Test nhanh bang API

Kiem tra server:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Neu chua init workspace bang CLI, co the init bang API:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/workspace/init `
  -ContentType "application/json" `
  -Body '{"model":"gpt-4.1-mini","embedding":"text-embedding-3-small","force":false}'
```

Neu workspace da co `settings.yaml` va muon tao lai config, doi `force` thanh `true`.

Them sample document:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/documents/sample?overwrite=true"
```

Index du lieu. Buoc nay se goi LLM va co the ton chi phi:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/index `
  -ContentType "application/json" `
  -Body '{"method":"fast","verbose":true,"cache":true}'
```

Query:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"question":"Who owns supplier integrations?","method":"local","response_type":"Single Sentence"}'
```

## Endpoint chinh

- `GET /health`: kiem tra server.
- `GET /workspace/status`: xem workspace, input files, output moi nhat.
- `POST /workspace/init`: goi `graphrag init --root workspace`.
- `POST /documents/sample`: tao file sample trong `workspace/input`.
- `POST /documents/text`: them file `.txt` tuy y vao `workspace/input`.
- `POST /index`: goi `graphrag index`.
- `POST /query`: goi `graphrag query`.

## Luu y

GraphRAG indexing co the dung nhieu token. Nen bat dau bang sample nho va method
`fast` truoc khi index bo tai lieu lon.
