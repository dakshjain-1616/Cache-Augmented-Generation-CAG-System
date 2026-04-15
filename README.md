# Cache-Augmented Generation (CAG) System

[![Built with NEO](https://img.shields.io/badge/Built%20with-NEO%20AI%20Agent-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMSAxNXYtNEg3bDUtOXY0aDRsLTUgOXoiLz48L3N2Zz4=)](https://heyneo.com)
[![NEO VS Code Extension](https://img.shields.io/visual-studio-marketplace/v/NeoResearchInc.heyneo?style=for-the-badge&label=NEO%20for%20VS%20Code&logo=visualstudiocode&logoColor=white&color=0078d7)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)

> This entire implementation — setup, debugging, GPU validation, and documentation — was built autonomously using **[NEO](https://heyneo.com)**, an autonomous AI agent. NEO wrote, debugged, and tested all the code, fixed 9 bugs across CUDA/Python/shell, and ran 11 GPU validation tests end-to-end. Try NEO in your IDE with the [VS Code extension](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo).

---

A **RAG-less** document QA system using long-context LLM + KV cache as a persistent document store. No embeddings, no vector DB, no chunking, no retrieval pipeline.

## Overview

Cache-Augmented Generation (CAG) is a novel approach to document question-answering that leverages the KV cache of transformer models as a persistent document store. Instead of:

- ❌ Chunking documents into small pieces
- ❌ Computing embeddings for each chunk
- ❌ Storing embeddings in a vector database
- ❌ Retrieving relevant chunks at query time

CAG simply:

- ✅ Loads documents into the model's context window
- ✅ Pre-computes and saves the KV cache
- ✅ Restores the cache instantly at query time
- ✅ Generates answers using the cached context

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CAG System                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Documents  │───▶│  Ingestion   │───▶│  KV Cache    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │         │
│                                                   ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Answer     │◀───│   Query      │◀───│   Restore    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. `setup.sh` — Environment Setup
- Detects GPU VRAM
- Clones and builds llama.cpp with CUDA support
- Downloads appropriate Qwen GGUF model based on available VRAM
- Creates directory structure and configuration

### 2. `start_server.sh` — LLM Server
- Launches llama-server with optimal settings
- Enables YaRN RoPE scaling for long contexts
- Configures TurboQuant KV compression (`turbo3`)
- Sets up Flash Attention and KV slot persistence

### 3. `src/ingest.py` — Document Ingestion
- Reads `.txt`, `.md`, or `.pdf` files (or stdin)
- Creates structured prompts for ingestion
- POSTs to llama-server to prefill KV cache
- Saves KV slot to `kv_slots/` via API
- Records metadata to `manifest.json`

### 4. `src/query.py` — Query Tool
- Restores KV slot by corpus ID
- Appends user question to cached context
- Streams LLM response to stdout
- Reports restore time and decode speed

### 5. `src/api_server.py` — REST API
- FastAPI server with endpoints:
  - `POST /ingest` — Ingest documents (async background job)
  - `GET /status/{job_id}` — Check ingestion status
  - `POST /query` — Query a corpus (streaming or non-streaming)
  - `GET /corpora` — List available corpora
  - `DELETE /corpora/{id}` — Delete a corpus
  - `GET /health` — Health check

### 6. `src/demo.py` — End-to-End Demo
- Downloads sample texts (Alice in Wonderland, Peter Pan)
- Ingests them via the REST API
- Runs 6 test questions and prints answers
- Prints performance statistics (restore time, decode speed)

## Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| **OS** | Linux (Ubuntu 20.04+ recommended) |
| **GPU** | NVIDIA GPU with CUDA 11.8+ — see VRAM table below |
| **VRAM** | 8 GB minimum (CPU fallback available but slow) |
| **RAM** | 16 GB+ recommended |
| **Disk** | 20 GB free (model ~16 GB + KV slots up to 3 GB each) |
| **Python** | 3.8+ |
| **Build tools** | `cmake`, `build-essential`, `git` (installed by `setup.sh`) |
| **Docker** | Optional — only needed for containerised deployment |

> **Windows / macOS:** not supported. llama.cpp CUDA builds require Linux. WSL2 on Windows may work but is untested.

### Local Installation

```bash
# 1. Clone and setup
git clone <repository>
cd cache-augmented-generation

# 2. Run setup (detects GPU, downloads model)
./setup.sh

# 3. Start llama-server
./start_server.sh

# 4. Ingest a document
python3 src/ingest.py my_document.txt --corpus-id my_doc --description "My document"

# 5. Query the document
python3 src/query.py my_doc "What is this document about?"
```

### Docker Deployment

```bash
# Build and start services
docker-compose -f docker/docker-compose.yml up -d

# Run demo
docker-compose -f docker/docker-compose.yml --profile demo run demo

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## Usage Examples

### Ingesting Documents

```bash
# From file
python3 src/ingest.py document.txt --corpus-id report_2024 --description "Annual report"

# From stdin
cat document.txt | python3 src/ingest.py --corpus-id report_2024

# From a directory of .txt/.md/.pdf files
python3 src/ingest.py --corpus-dir ./my_docs --corpus-id collection
```

### Querying Documents

```bash
# Basic query
python3 src/query.py report_2024 "What were the main findings?"

# With custom parameters
python3 src/query.py report_2024 "Summarize the conclusion" --max-tokens 256 --temperature 0.5

# List available corpora
python3 src/query.py --list
```

### Using the API

```bash
# Start API server
python3 src/api_server.py --port 8000

# Optional: enable API key auth
CAG_API_KEY=mysecretkey python3 src/api_server.py

# Ingest via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mysecretkey" \
  -d '{
    "documents": ["Document content here..."],
    "corpus_id": "my_corpus",
    "description": "My document"
  }'

# Poll job status
curl http://localhost:8000/status/<job_id>

# Query via API (streaming)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mysecretkey" \
  -d '{"corpus_id": "my_corpus", "question": "What is this about?", "stream": true}'

# List corpora
curl http://localhost:8000/corpora

# Delete a corpus
curl -X DELETE http://localhost:8000/corpora/my_corpus \
  -H "X-API-Key: mysecretkey"
```

## Performance

CAG provides significant performance advantages over traditional RAG:

| Metric | Traditional RAG | CAG |
|--------|----------------|-----|
| Ingestion Time | O(n) embedding + index | O(n) prefill once |
| Query Latency | Retrieval + Generation | Instant restore + Generation |
| Context Quality | Chunked, potentially incomplete | Full document context |
| Memory Usage | Vector DB + Model | Model KV cache only |

Measured performance from original demo (NVIDIA L4 24GB, Qwen3.5-35B-A3B Q3_K_M):
- **Prefill**: ~68 minutes for 905K tokens (1M context)
- **Slot restore**: ~2.3 seconds (vs 68-min cold prefill)
- **Decode at 1M context**: ~9 tok/s
- **Decode at 262K context**: ~20 tok/s
- **KV compression**: 23GB → 4GB via Turbo3 (4.3x)
- **VRAM utilisation**: ~96% on 24GB

> See `GPU_TESTING.md` for full GPU validation checklist.

## Model Selection

The system automatically selects models based on available VRAM:

| VRAM | Model | Context Size | Notes |
|------|-------|--------------|-------|
| 24GB+ | Qwen3.5-35B-A3B Q3_K_M | **1M tokens** | Target config — matches original demo |
| 16-24GB | Qwen2.5-7B Q4_K_M | 32K | Good quality, fast |
| 8-16GB | Qwen2.5-7B Q4_0 | 16K | Lighter quantisation |
| CPU / <8GB | Qwen2.5-3B Q4_K_M | 4K | Fallback — demonstrates workflow only |

## Configuration

Environment variables (set in `.env`):

```bash
GPU_AVAILABLE=true
GPU_VRAM_MB=16384
MODEL_PATH=/app/models/qwen2.5-7b-instruct-q4_k_m.gguf
MODEL_NAME=qwen2.5-7b-instruct-q4_k_m.gguf
CTX_SIZE=32768
LLAMA_CPP_PORT=8080
API_PORT=8000
KV_SLOTS_DIR=/app/kv_slots
CORPORA_DIR=/app/corpora
```

## API Reference

### POST /ingest
Ingest documents into KV cache.

**Request:**
```json
{
  "documents": ["string"],
  "corpus_id": "string (optional)",
  "description": "string (optional)"
}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "corpus_id": "string",
  "message": "Ingestion started"
}
```

### GET /status/{job_id}
Check ingestion job status.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "corpus_id": "string",
  "result": {...},
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

### DELETE /corpora/{corpus_id}
Delete a corpus and its KV slot file. Requires `X-API-Key` header if `CAG_API_KEY` env var is set.

---

### POST /query
Query a corpus. Requires `X-API-Key` header if `CAG_API_KEY` env var is set.

**Request:**
```json
{
  "corpus_id": "string",
  "question": "string",
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": true
}
```

**Response:**
```json
{
  "corpus_id": "string",
  "question": "string",
  "answer": "string",
  "tokens_generated": 100,
  "restore_time_ms": 50.0,
  "decode_time_s": 2.5,
  "tokens_per_sec": 40.0,
  "ms_per_token": 25.0,
  "timestamp": "timestamp"
}
```

### GET /corpora
List all available corpora.

**Response:**
```json
[
  {
    "corpus_id": "string",
    "description": "string",
    "slot_id": "string",
    "prompt_length": 10000,
    "prefill_ms": 5000.0,
    "total_time_s": 10.0,
    "timestamp": "timestamp"
  }
]
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "llama_server": "healthy",
  "timestamp": "timestamp"
}
```

## Directory Structure

```
.
├── setup.sh              # Environment setup — builds llama.cpp, downloads model
├── start_server.sh       # Launch llama-server with CAG-optimised flags
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .env                  # Environment configuration (auto-generated by setup.sh)
├── manifest.json         # Corpus metadata (runtime, not committed)
├── jobs.json             # Ingestion job state (runtime, not committed)
│
├── src/                  # Python source
│   ├── api_server.py     # FastAPI REST API
│   ├── ingest.py         # CLI: ingest a document
│   ├── query.py          # CLI: query a corpus
│   └── demo.py           # End-to-end demo (Alice + Peter Pan)
│
├── docker/               # Container setup
│   ├── Dockerfile        # Multi-stage CUDA image
│   └── docker-compose.yml
│
├── docs/                 # Documentation
│   ├── REPORT.md         # Full GPU validation report
│   └── GPU_TESTING.md    # GPU test checklist
│
├── models/               # GGUF model weights (not committed)
├── kv_slots/             # Saved KV cache .bin files (not committed)
├── corpora/              # Raw document storage (not committed)
└── logs/                 # Runtime logs (not committed)
```

## Limitations

Understanding these upfront will save you debugging time:

| Limitation | Detail |
|---|---|
| **Linux + NVIDIA only** | The TurboQuant CUDA kernels require a Linux NVIDIA GPU. No Windows, no macOS, no AMD/Apple Silicon. |
| **Long first prefill** | A 900K-token document takes ~24 min to prefill on an A6000. This is a one-time cost per document — subsequent queries restore in ~1–2 seconds. |
| **VRAM gating** | Below 24 GB VRAM you get a smaller model with a shorter context window (see model selection table). Sub-8 GB falls back to CPU, which is very slow. |
| **One slot, one document** | The current implementation uses a single llama.cpp slot (slot 0). Only one corpus can be active at a time. Switching corpora requires a slot restore (~1 sec). |
| **Lost-in-the-middle** | With YaRN 4× context extension (262K → 1M tokens), the model attends strongly to the beginning and end of the document but can miss content in the middle. Works best for targeted questions about known sections. |
| **Build time** | First-time `./setup.sh` takes ~35 minutes to compile hundreds of CUDA Flash Attention kernel templates. This is a one-time cost. |
| **No streaming in CLI** | `query.py` streams tokens to stdout. The REST API supports both streaming (`stream: true`) and non-streaming responses. |
| **Model download requires HF token** | The Qwen3.5-35B model is gated on HuggingFace. You need a free HF account and access token for the 24 GB+ path. Smaller models are public. |

## Troubleshooting

### Server won't start
```bash
# Check GPU availability
nvidia-smi

# Check port availability
lsof -i :8080
lsof -i :8000

# View logs
tail -f server.log
```

### Ingestion fails
```bash
# Check server health
curl http://localhost:8080/health

# Verify model exists
ls -la models/

# Check manifest
jq . manifest.json
```

### Query returns empty
```bash
# Check corpus exists
python query.py --list

# Verify slot file
ls -la kv_slots/

# Check server slots
curl http://localhost:8080/slots
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details.

## GPU Testing

All 11 GPU validation tests have been run and passed on an NVIDIA RTX A6000 (48 GB).
See **`docs/REPORT.md`** for full results, and **`docs/GPU_TESTING.md`** for the test checklist.

Key validated results:
- TurboQuant `turbo3` KV compression: 23 GB → 4 GB at 1M context ✅
- YaRN 4× context extension: 262K → 1,048,576 tokens ✅
- War and Peace (922K tokens) prefilled in **24 minutes**, restored in **1.2 seconds** ✅
- Decode speed: **~100 tok/s** at full 1M context ✅

## Acknowledgments

- [NEO](https://heyneo.com) — Autonomous AI agent that built, debugged, and validated this entire implementation. Also available as a [VS Code extension](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo).
- [llama-cpp-turboquant](https://github.com/atomicmilkshake/llama-cpp-turboquant) — TurboQuant fork with KV compression
- [Han Xiao / Jina AI](https://www.linkedin.com/in/hxiao87) — Original CAG concept and demo
- [Qwen](https://github.com/QwenLM/Qwen) — Qwen3.5-35B-A3B model
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework

## Citation

If you use CAG in your research, please cite:

```bibtex
@software{cag2024,
  title = {Cache-Augmented Generation (CAG) System},
  author = {CAG Contributors},
  year = {2024},
  url = {https://github.com/yourusername/cag}
}
```

## Contact

For questions or support, please open an issue on GitHub.
