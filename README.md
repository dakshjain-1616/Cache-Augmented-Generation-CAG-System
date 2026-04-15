# Cache-Augmented Generation (CAG) System

[![Built with NEO](https://img.shields.io/badge/Built%20with-NEO%20AI%20Agent-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMSAxNXYtNEg3bDUtOXY0aDRsLTUgOXoiLz48L3N2Zz4=)](https://heyneo.com)
[![NEO for VS Code](https://img.shields.io/badge/VS%20Code-Install%20Extension-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)

> This entire implementation — setup, debugging, GPU validation, and documentation — was built autonomously using **[NEO](https://heyneo.com)**, an autonomous AI agent. NEO wrote, debugged, and tested all the code, fixed 9 bugs across CUDA/Python/shell, and ran 11 GPU validation tests end-to-end. Try NEO in your IDE with the [VS Code extension](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo).

---

A **RAG-less** document QA system that loads entire documents into an LLM's KV cache once, saves it to disk, and restores it instantly before every query. No embeddings, no vector DB, no chunking.

---

## What is CAG?

Traditional RAG breaks documents into chunks, embeds them, stores them in a vector DB, and retrieves the most relevant chunks at query time. This works but loses context — the model only ever sees fragments.

Cache-Augmented Generation takes a different approach: load the **entire document** into the model's context window, let the model prefill the full KV cache, save that cache to disk, and restore it in seconds before every query. The model sees everything, every time.

| | Traditional RAG | Cache-Augmented Generation |
|---|---|---|
| Context quality | Chunked fragments | Full document, always |
| Query latency | Retrieval + generation | Instant restore + generation |
| Setup complexity | Embeddings + vector DB | One prefill, done |
| Repeated queries | Re-retrieve every time | Restore once per restart |

The original concept was demonstrated by **[Han Xiao](https://www.linkedin.com/in/hxiao87)** showing 1M-token inference on a single 24 GB GPU — the entire book in context at once.

---

## How It Works

**Step 1 — Ingest (done once per document)**

The document is wrapped in a structured prompt and sent to `llama-server`. The model runs a full prefill pass, loading every token into the KV cache. This takes time — proportional to document size — but only happens once. The KV cache is then saved to a `.bin` file on disk.

**Step 2 — Query (instant, repeatable)**

Before each query, the saved `.bin` file is restored into `llama-server`'s KV cache in ~1 second. The user's question is appended and the model generates an answer with full document context active. No re-reading, no re-embedding.

**Step 3 — Persistence**

KV slots survive server restarts. Kill the server, restart it, and your next query restores the cache from disk just as fast. The 24-minute prefill for War and Peace only needs to happen once — ever.

---

## Validated Results

All 11 GPU tests were run on an **NVIDIA RTX A6000 (48 GB VRAM)** with **Qwen3.5-35B-A3B Q3_K_M** at **1,048,576 token context**.

### Performance

| Metric | Reference (L4 24 GB) | Our Results (A6000 48 GB) |
|---|---|---|
| Cold prefill — War & Peace (922K tokens) | ~68 minutes | **24.3 minutes** |
| KV slot restore from disk | ~2.3 seconds | **~1.2 seconds** |
| Decode speed at 1M context | ~9 tok/s | **~100 tok/s** |
| KV cache size at 1M context | 23 GB (f16) | **4 GB (turbo3)** — 5.75× compression |
| VRAM used (model + KV cache) | ~96% of 24 GB | **43% of 48 GB** |

### Test Results Summary

| Test | Result |
|---|---|
| TurboQuant build — `turbo2/3/4` cache types available | ✅ Pass |
| KV compression — 4 GB at 1M context (vs 23 GB f16) | ✅ Pass |
| YaRN context extension — 262K → 1,048,576 tokens | ✅ Pass |
| Slot save/restore timing | ✅ Pass |
| VRAM profile at each stage | ✅ Pass |
| Flash Attention active | ✅ Pass |
| End-to-end demo — Alice in Wonderland + Peter Pan, 6/6 queries answered | ✅ Pass |
| Concurrent query handling — 5 simultaneous requests, no corruption | ✅ Pass |
| War & Peace stress test — 922K tokens, correct answers on characters & battles | ✅ Pass |
| API authentication — key auth working | ✅ Pass |
| Slot persistence across server restarts | ✅ Pass |

### Sample Queries — War and Peace (922K tokens, 1M context)

| Question | Result |
|---|---|
| "Who is Pierre Bezukhov?" | Correct, detailed answer |
| "What happened at the Battle of Borodino?" | Correct, detailed answer |
| "How does the novel end?" | Partial — lost-in-middle at extreme context depth (see Limitations) |

### Demo Run — Alice in Wonderland + Peter Pan

| Metric | Result |
|---|---|
| Documents ingested | 2/2 |
| Queries answered correctly | 6/6 |
| Average slot restore time | 0.1 ms |
| Average decode speed | 103 tok/s |
| Average latency | 9.7 ms/token |
| OOM errors | None |

---

## Quick Start

**Prerequisites:** Linux, NVIDIA GPU (8 GB+ VRAM), Python 3.8+

```bash
# 1. Build llama.cpp + download model (one-time, ~35 min)
./setup.sh

# 2. Start the LLM server
./start_server.sh

# 3. Start the API server
python3 src/api_server.py

# 4. Ingest a document
python3 src/ingest.py my_document.txt --corpus-id my_doc

# 5. Query it
python3 src/query.py my_doc "What is this document about?"
```

That's it. After step 4, the KV cache is saved to `kv_slots/my_doc.bin`. Every future query restores it instantly, and it survives server restarts.

---

## Model Selection

`setup.sh` auto-detects GPU VRAM and picks the right model:

| VRAM | Model | Context | Notes |
|---|---|---|---|
| 24 GB+ | Qwen3.5-35B-A3B Q3_K_M | **1M tokens** | Full CAG — matches original demo |
| 16–24 GB | Qwen2.5-7B Q4_K_M | 32K | Good quality |
| 8–16 GB | Qwen2.5-7B Q4_0 | 16K | Lighter |
| CPU / <8 GB | Qwen2.5-3B Q4_K_M | 4K | Demo only — very slow |

> The 24 GB+ path uses `unsloth/Qwen3.5-35B-A3B-GGUF` on HuggingFace and requires a free HF account + access token.

---

## REST API

Start the API server with `python3 src/api_server.py --port 8000` (optionally set `CAG_API_KEY` env var to enable key auth).

| Endpoint | Method | Description |
|---|---|---|
| `/ingest` | POST | Ingest documents — returns a `job_id` immediately, runs in background |
| `/status/{job_id}` | GET | Poll ingestion job status (`pending` → `processing` → `completed`) |
| `/query` | POST | Query a corpus — supports `stream: true/false` |
| `/corpora` | GET | List all ingested corpora |
| `/corpora/{id}` | DELETE | Delete a corpus and its KV slot file |
| `/health` | GET | Health check (no auth required) |

Full API docs available at `http://localhost:8000/docs` when the server is running.

---

## Directory Structure

```
.
├── setup.sh              # Builds llama.cpp, downloads model
├── start_server.sh       # Launches llama-server with CAG flags
├── requirements.txt
├── src/
│   ├── api_server.py     # FastAPI REST API
│   ├── ingest.py         # CLI: ingest a document
│   ├── query.py          # CLI: query a corpus
│   └── demo.py           # End-to-end demo
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── REPORT.md         # Full GPU validation report with all 11 test results
│   └── GPU_TESTING.md    # GPU test checklist
├── models/               # GGUF weights (not committed)
├── kv_slots/             # Saved KV cache .bin files (not committed)
└── logs/                 # Runtime logs (not committed)
```

---

## Limitations

| Limitation | Detail |
|---|---|
| **Linux + NVIDIA only** | TurboQuant CUDA kernels require Linux + NVIDIA. No Windows, macOS, or AMD. |
| **Long first prefill** | 900K tokens takes ~24 min on an A6000. One-time cost — all future queries restore in ~1 sec. |
| **VRAM gating** | Below 24 GB you get a smaller model with shorter context (see table above). |
| **One active corpus** | Single llama.cpp slot (slot 0). Switching corpora costs ~1 sec restore. |
| **Lost-in-the-middle** | YaRN 4× extrapolation attends strongly to document start/end. Mid-document content can be missed on very long docs. |
| **Build time** | First `./setup.sh` takes ~35 min to compile CUDA kernels. One-time cost. |
| **HF token for large model** | Qwen3.5-35B is gated. Free HF account + token required. Smaller models are public. |

---

## Troubleshooting

**Server won't start** — check `nvidia-smi` to confirm GPU is visible, check ports 8080/8000 aren't in use, tail `logs/server.log`.

**Ingestion fails** — confirm `llama-server` is healthy at `http://localhost:8080/health`, verify the model file exists in `models/`.

**Query returns empty** — run `python3 src/query.py --list` to confirm the corpus exists, check `kv_slots/` for the `.bin` file.

---

## Acknowledgments

- [NEO](https://heyneo.com) — Autonomous AI agent that built, debugged, and validated this entire implementation
- [llama-cpp-turboquant](https://github.com/atomicmilkshake/llama-cpp-turboquant) — TurboQuant KV compression fork
- [Han Xiao / Jina AI](https://www.linkedin.com/in/hxiao87) — Original CAG concept and demo
- [Qwen / Alibaba](https://github.com/QwenLM/Qwen) — Qwen3.5-35B-A3B MoE model
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
