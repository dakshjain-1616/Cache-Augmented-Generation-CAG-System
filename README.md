# Cache-Augmented Generation (CAG) System

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

### 1. `setup.sh` - Environment Setup
- Detects GPU VRAM
- Clones and builds llama.cpp with CUDA support
- Downloads appropriate Qwen GGUF model based on available VRAM
- Creates directory structure and configuration

### 2. `start_server.sh` - LLM Server
- Launches llama-server with optimal settings
- Enables YaRN RoPE scaling for long contexts
- Configures flash attention for efficiency
- Sets up KV slot persistence

### 3. `ingest.py` - Document Ingestion
- Reads documents from files or stdin
- Creates structured prompts for ingestion
- POSTs to llama-server to prefill KV cache
- Saves KV slot via API
- Records metadata to manifest.json

### 4. `query.py` - Query Tool
- Restores KV slot by corpus_id
- Appends user question to cached context
- Streams LLM response
- Reports restore time and decode speed

### 5. `api_server.py` - REST API
- FastAPI server with endpoints:
  - `POST /ingest` - Ingest documents
  - `GET /status/{job_id}` - Check ingestion status
  - `POST /query` - Query a corpus
  - `GET /corpora` - List available corpora
  - `GET /health` - Health check

### 6. `demo.py` - End-to-End Demo
- Downloads sample texts (Alice in Wonderland, Peter Pan)
- Ingests documents into KV cache
- Runs test questions
- Prints performance statistics

## Quick Start

### Prerequisites

- Linux with CUDA support (for GPU acceleration)
- Python 3.8+
- Docker and Docker Compose (optional)

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
python ingest.py my_document.txt --corpus-id my_doc --description "My document"

# 5. Query the document
python query.py my_doc "What is this document about?"
```

### Docker Deployment

```bash
# Build and start services
docker-compose up -d

# Run demo
docker-compose --profile demo run demo

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Usage Examples

### Ingesting Documents

```bash
# From file
python ingest.py document.txt --corpus-id report_2024 --description "Annual report"

# From stdin
cat document.txt | python ingest.py --corpus-id report_2024

# Multiple documents
python ingest.py doc1.txt doc2.txt doc3.txt --corpus-id collection
```

### Querying Documents

```bash
# Basic query
python query.py report_2024 "What were the main findings?"

# With custom parameters
python query.py report_2024 "Summarize the conclusion" --max-tokens 256 --temperature 0.5

# List available corpora
python query.py --list
```

### Using the API

```bash
# Start API server
python api_server.py --port 8000

# Optional: enable API key auth
CAG_API_KEY=mysecretkey python api_server.py

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
├── setup.sh              # Environment setup
├── start_server.sh      # Launch llama-server
├── ingest.py            # Document ingestion
├── query.py             # Query tool
├── api_server.py        # FastAPI server
├── demo.py              # End-to-end demo
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-service setup
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .env                 # Environment configuration
├── manifest.json        # Corpus metadata
├── kv_slots/            # KV cache storage
├── corpora/             # Document storage
└── models/              # GGUF model files
```

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

Before deploying, validate on real GPU hardware using the checklist in **`GPU_TESTING.md`**.

Critical GPU-only steps not yet validated:
- TurboQuant `--cache-type-k q3_K --cache-type-v q3_K` flags (add to `start_server.sh`)
- Actual VRAM utilisation at 1M context
- YaRN quality at 905K token depth
- Slot restore timing

## Acknowledgments

- [llama-cpp-turboquant](https://github.com/atomicmilkshake/llama-cpp-turboquant) - TurboQuant fork with KV compression
- [Han Xiao / Jina AI](https://www.linkedin.com/in/hxiao87) - Original CAG concept and demo
- [Qwen](https://github.com/QwenLM/Qwen) - Qwen3.5-35B-A3B model
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework

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
