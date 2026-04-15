# GPU Testing Checklist

This file documents everything that **must be validated on actual GPU hardware** before the system is considered production-ready. All items below cannot be tested without a CUDA-capable GPU (ideally NVIDIA L4 24GB as used in the original Han Xiao demo).

---

## Hardware Requirements

| Component | Minimum | Target (original demo) |
|---|---|---|
| GPU | Any CUDA GPU with 8GB+ VRAM | NVIDIA L4 24GB |
| VRAM | 8GB (small model) | 24GB (Qwen3.5-35B-A3B) |
| RAM | 16GB | 32GB+ |
| Disk | 50GB free | 100GB+ (models + KV slots) |
| CUDA | 11.8+ | 12.1 |

---

## Test 1 — TurboQuant Build Verification

**What:** Confirm the TurboQuant fork builds correctly with CUDA and Flash Attention enabled.

**How to test:**
```bash
./setup.sh 2>&1 | tee setup_output.log

# After build, check for turbo support in the binary
./llama-cpp-turboquant/build/bin/llama-server --help | grep -i turbo
./llama-cpp-turboquant/build/bin/llama-server --help | grep -i cache-type
```

**Expected:**
- Build completes without errors
- `--cache-type-k` and `--cache-type-v` flags are available
- No missing CUDA library errors

**If it fails:**
- Check CUDA version: `nvcc --version`
- Check CMake version: `cmake --version` (need 3.14+)
- Verify `-DGGML_CUDA=ON -DLLAMA_FLASH_ATTN=ON` flags were passed

---

## Test 2 — TurboQuant KV Compression (The Critical One)

**What:** Validate that Turbo3 KV cache compression actually works — this is what makes 1M context fit in 24GB VRAM.

**Add these flags to `start_server.sh` before testing:**
```bash
--cache-type-k q3_K \
--cache-type-v q3_K \
```

**How to test:**
```bash
# Start server with TurboQuant flags
./start_server.sh

# In another terminal, watch VRAM usage
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Prefill a small test corpus
echo "This is a test document." | python ingest.py --corpus-id test_turbo

# Check VRAM: with Turbo3 it should be significantly lower than without
```

**Expected VRAM at 1M context (905K tokens prefilled):**
- Without TurboQuant: ~23GB (won't fit on 24GB)
- With Turbo3 (`q3_K`): ~4GB for KV cache + ~15GB model = ~19GB total ✅

**Record:** VRAM before prefill / during prefill / after prefill

**If compression isn't working:**
- VRAM will spike beyond GPU capacity
- Server will OOM crash
- Verify `--cache-type-k q3_K --cache-type-v q3_K` flags are in the launch command

---

## Test 3 — YaRN Context Extension

**What:** Validate that YaRN RoPE scaling actually extends context from 131K → 1M tokens without degrading quality.

**How to test:**
```bash
# Start server (flags already set: --rope-scaling yarn --rope-scale 4.0 --yarn-orig-ctx 131072)
./start_server.sh

# Check server accepted the context size
curl http://localhost:8080/health | jq .
curl http://localhost:8080/props | jq '.total_slots, .n_ctx'
```

**Expected:**
- `n_ctx` reports 1048576 (or your configured CTX_SIZE)
- Server starts without errors about context size
- No warnings about RoPE scaling

**Extended test — quality at depth:**
```bash
# Ingest a large document (200K+ tokens)
python ingest.py large_document.txt --corpus-id yarn_test

# Ask a question about content near position 100K (middle)
python query.py --corpus-id yarn_test --question "What is discussed in the middle section?"

# Ask about content near position 200K (end)
python query.py --corpus-id yarn_test --question "What is the conclusion?"
```

**Record:** Does the model correctly answer questions about content in the middle vs end of a long document? Note any hallucinations.

**Known limitation:** YaRN 4x extrapolation causes "lost in the middle" — model attends primarily to first and last tokens. Document any hallucinations observed.

---

## Test 4 — KV Slot Save/Restore Timing

**What:** Validate the core CAG mechanism — prefill once, restore instantly.

**How to test:**
```bash
# Ingest a large corpus and time the prefill
time python ingest.py large_document.txt --corpus-id timing_test

# Then time the restore + query
time python query.py --corpus-id timing_test --question "Summarize this document."
```

**Target timings (from original demo on L4):**
| Operation | Target | Acceptable |
|---|---|---|
| Cold prefill (1M tokens) | ~68 minutes | < 2 hours |
| Slot restore from disk | ~2.3 seconds | < 10 seconds |
| Decode speed (1M context) | ~9 tok/s | > 5 tok/s |
| Decode speed (262K context) | ~20 tok/s | > 10 tok/s |

**Record actual timings in the table above.**

---

## Test 5 — VRAM Utilization Profile

**What:** Verify VRAM usage matches expected profile across the full lifecycle.

**How to test:**
```bash
# Terminal 1: continuous VRAM monitoring
nvidia-smi dmon -s mu -d 5 | tee vram_log.txt

# Terminal 2: run the full lifecycle
./start_server.sh &
sleep 10
python ingest.py large_document.txt --corpus-id vram_test
python query.py --corpus-id vram_test --question "What is this about?"
```

**Expected VRAM profile for Qwen3.5-35B-A3B Q3_K_M on 24GB:**
| Stage | Expected VRAM |
|---|---|
| Server idle (model loaded) | ~15.2GB (model weights) |
| During 1M prefill | ~19-21GB (model + KV cache building) |
| After prefill (slot saved) | ~15.2GB (KV evicted from GPU) |
| During query (slot restored) | ~19-21GB |
| Target max utilization | ~96% of 24GB = ~23GB |

**Record actual values at each stage.**

---

## Test 6 — Flash Attention Validation

**What:** Confirm Flash Attention is active and providing memory savings.

**How to test:**
```bash
# Check Flash Attention is loaded in server logs
./start_server.sh 2>&1 | grep -i "flash\|attention"

# Compare memory usage with and without flash attention
# Run with --flash-attn (current config) vs without
```

**Expected:**
- Server log shows Flash Attention enabled
- VRAM usage is lower than without Flash Attention
- No CUDA errors related to attention

---

## Test 7 — End-to-End Demo on Real Hardware

**What:** Run the full demo pipeline on GPU and compare results to expected.

**How to test:**
```bash
# Start both servers
./start_server.sh &
sleep 15
python api_server.py &
sleep 5

# Run demo
python demo.py 2>&1 | tee demo_output.log
```

**Expected:**
- Both Alice in Wonderland and Peter Pan are ingested successfully
- All 3 test questions per corpus return coherent answers
- No OOM errors
- Performance stats printed match target timings from Test 4

---

## Test 8 — Concurrent Query Handling

**What:** Validate that the `asyncio.Lock()` on slot 0 correctly serialises concurrent queries (no slot corruption).

**How to test:**
```bash
# After ingesting a corpus:
python ingest.py document.txt --corpus-id concurrency_test

# Fire 5 concurrent queries
for i in {1..5}; do
    curl -s -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"corpus_id": "concurrency_test", "question": "What is this document about?", "stream": false}' &
done
wait

# All 5 should return valid answers, none should be empty or corrupted
```

**Expected:** All 5 responses contain valid answers. No 500 errors. No corrupted/empty responses.

---

## Test 9 — Large Document Ingestion (Stress Test)

**What:** Test ingestion of a genuinely large corpus approaching 1M tokens.

**How to test:**
```bash
# Download a large public domain book collection (~1M tokens)
python -c "
import urllib.request
url = 'https://www.gutenberg.org/files/2600/2600-0.txt'  # War and Peace ~580K tokens
urllib.request.urlretrieve(url, 'war_and_peace.txt')
print('Downloaded')
"

python ingest.py war_and_peace.txt --corpus-id war_and_peace

# Ask questions about specific chapters
python query.py --corpus-id war_and_peace --question "Who is Pierre Bezukhov?"
python query.py --corpus-id war_and_peace --question "What happens at the Battle of Borodino?"
python query.py --corpus-id war_and_peace --question "How does the novel end?"
```

**Record:**
- Token count estimated by ingest.py
- Actual prefill time
- Answer quality for questions about beginning / middle / end of book
- Any "lost in the middle" hallucinations noted

---

## Test 10 — API Authentication

**What:** Validate the optional API key auth works correctly.

**How to test:**
```bash
# Start with API key enabled
CAG_API_KEY=mysecretkey python api_server.py &

# Should FAIL (no key)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": ["test"], "corpus_id": "test"}'
# Expected: 403 Forbidden

# Should SUCCEED (correct key)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mysecretkey" \
  -d '{"documents": ["test"], "corpus_id": "test"}'
# Expected: 200 with job_id

# Health check should work without key
curl http://localhost:8000/health
# Expected: 200
```

---

## Test 11 — Slot Persistence Across Server Restarts

**What:** Verify that a saved KV slot can be restored after restarting llama-server.

**How to test:**
```bash
# Ingest corpus
python ingest.py document.txt --corpus-id persist_test

# Kill llama-server
pkill llama-server

# Restart llama-server
./start_server.sh &
sleep 10

# Query (should restore slot from disk)
python query.py --corpus-id persist_test --question "What is this about?"
```

**Expected:** Query succeeds after restart. Slot restored from disk without re-prefilling.

---

## Results Recording Template

Fill this in after testing on GPU:

```
Date: _______________
GPU: _______________
VRAM: _______________ GB
CUDA version: _______________
Driver version: _______________
Model: _______________

Test 1 — TurboQuant Build:    PASS / FAIL
Test 2 — KV Compression:      PASS / FAIL   VRAM at 1M ctx: ___ GB
Test 3 — YaRN Extension:      PASS / FAIL   Lost-in-middle observed: YES / NO
Test 4 — Slot Timing:         PASS / FAIL   Prefill: ___min  Restore: ___s  Decode: ___tok/s
Test 5 — VRAM Profile:        PASS / FAIL   Peak VRAM: ___ GB  Utilization: ___%
Test 6 — Flash Attention:     PASS / FAIL
Test 7 — E2E Demo:            PASS / FAIL
Test 8 — Concurrency:         PASS / FAIL
Test 9 — Large Doc (580K tok):PASS / FAIL   Hallucinations noted: ___
Test 10 — API Auth:           PASS / FAIL
Test 11 — Persistence:        PASS / FAIL

Notes:
_______________________________________________
_______________________________________________
```

---

## What to Add to start_server.sh Before GPU Testing

Before running any tests, add the TurboQuant KV cache flags to `start_server.sh`. Find the `exec "${SERVER_BIN}"` block and add:

```bash
--cache-type-k q3_K \
--cache-type-v q3_K \
```

These are intentionally NOT in the current config because they can only be validated on real GPU hardware. Once confirmed working, they should be made the default.
