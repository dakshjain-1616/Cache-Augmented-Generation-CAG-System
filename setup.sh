#!/bin/bash
# Cache-Augmented Generation (CAG) System Setup Script
# Detects GPU VRAM, clones llama-cpp-turboquant, builds with CUDA support, downloads appropriate Qwen model based on VRAM, creates directory structure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
MODELS_DIR="${PROJECT_DIR}/models"
KV_SLOTS_DIR="${PROJECT_DIR}/kv_slots"
CORPORA_DIR="${PROJECT_DIR}/corpora"

echo "========================================"
echo "CAG System Setup"
echo "========================================"

# Create directory structure
echo "[1/5] Creating directory structure..."
mkdir -p "${MODELS_DIR}" "${KV_SLOTS_DIR}" "${CORPORA_DIR}"

# Detect GPU VRAM
echo "[2/5] Detecting GPU VRAM..."
GPU_VRAM_MB=0
GPU_AVAILABLE=false

if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
    if [ -n "$GPU_MEM" ]; then
        GPU_VRAM_MB=$GPU_MEM
        GPU_VRAM_GB=$((GPU_VRAM_MB / 1024))
        GPU_AVAILABLE=true
        echo "    ✓ GPU detected: ${GPU_VRAM_GB}GB VRAM"
    else
        echo "    ⚠ nvidia-smi found but no GPU detected"
    fi
else
    echo "    ⚠ nvidia-smi not found, assuming CPU-only mode"
fi

# Determine model size based on VRAM
if [ "$GPU_AVAILABLE" = true ]; then
    if [ "$GPU_VRAM_GB" -ge 24 ]; then
        MODEL_URL="https://huggingface.co/bartowski/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q3_K_M.gguf"
        MODEL_NAME="Qwen3.5-35B-A3B-Q3_K_M.gguf"
        CTX_SIZE=1048576
        echo "    → Selected model: Qwen3.5-35B-A3B (Q3_K_M) — 1M context on 24GB VRAM"
    elif [ "$GPU_VRAM_GB" -ge 16 ]; then
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"
        MODEL_NAME="qwen2.5-7b-instruct-q4_k_m.gguf"
        CTX_SIZE=32768
        echo "    → Selected model: Qwen2.5-7B (Q4_K_M) for 16GB+ VRAM"
    elif [ "$GPU_VRAM_GB" -ge 8 ]; then
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_0.gguf"
        MODEL_NAME="qwen2.5-7b-instruct-q4_0.gguf"
        CTX_SIZE=16384
        echo "    → Selected model: Qwen2.5-7B (Q4_0) for 8GB+ VRAM"
    else
        MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
        MODEL_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"
        CTX_SIZE=8192
        echo "    → Selected model: Qwen2.5-3B (Q4_K_M) for <8GB VRAM"
    fi
else
    # CPU fallback
    MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    MODEL_NAME="qwen2.5-3b-instruct-q4_k_m.gguf"
    CTX_SIZE=4096
    echo "    → Selected model: Qwen2.5-3B (Q4_K_M) for CPU mode"
fi

# Save configuration
cat > "${PROJECT_DIR}/.env" << EOF
GPU_AVAILABLE=${GPU_AVAILABLE}
GPU_VRAM_MB=${GPU_VRAM_MB}
MODEL_PATH=${MODELS_DIR}/${MODEL_NAME}
MODEL_NAME=${MODEL_NAME}
CTX_SIZE=${CTX_SIZE}
LLAMA_CPP_PORT=8080
API_PORT=8000
KV_SLOTS_DIR=${KV_SLOTS_DIR}
CORPORA_DIR=${CORPORA_DIR}
EOF

echo "    ✓ Configuration saved to .env"

# Clone and build llama-cpp-turboquant
echo "[3/5] Cloning llama-cpp-turboquant..."
LLAMA_CPP_DIR="${PROJECT_DIR}/llama-cpp-turboquant"

if [ -d "${LLAMA_CPP_DIR}" ]; then
    echo "    ⚠ llama-cpp-turboquant already exists, updating..."
    cd "${LLAMA_CPP_DIR}"
    git pull
else
    git clone https://github.com/atomicmilkshake/llama-cpp-turboquant "${LLAMA_CPP_DIR}"
    cd "${LLAMA_CPP_DIR}"
fi

# Build with appropriate settings
echo "[4/5] Building llama.cpp..."
mkdir -p build && cd build

CMAKE_FLAGS="-DLLAMA_BUILD_SERVER=ON"

if [ "$GPU_AVAILABLE" = true ]; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DGGML_CUDA=ON -DLLAMA_FLASH_ATTN=ON -DCMAKE_CUDA_ARCHITECTURES=all"
    echo "    → Building with CUDA support"
else
    CMAKE_FLAGS="${CMAKE_FLAGS} -DLLAMA_CUDA=OFF"
    echo "    → Building for CPU only"
fi

cmake .. ${CMAKE_FLAGS}
make -j$(nproc)

echo "    ✓ Build complete"

# Download model
echo "[5/5] Downloading Qwen GGUF model..."
MODEL_PATH="${MODELS_DIR}/${MODEL_NAME}"

if [ -f "${MODEL_PATH}" ]; then
    echo "    ✓ Model already exists: ${MODEL_NAME}"
else
    echo "    → Downloading ${MODEL_NAME}..."
    curl -L --progress-bar -o "${MODEL_PATH}" "${MODEL_URL}"
    echo "    ✓ Model downloaded"
fi

# Create manifest file if not exists
MANIFEST_FILE="${PROJECT_DIR}/manifest.json"
if [ ! -f "${MANIFEST_FILE}" ]; then
    echo '{"corpora": [], "version": "1.0"}' > "${MANIFEST_FILE}"
    echo "    ✓ Created manifest.json"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Context Size: ${CTX_SIZE}"
echo "GPU Available: ${GPU_AVAILABLE}"
echo ""
echo "Next steps:"
echo "  1. Run: ./start_server.sh"
echo "  2. Run: python3 src/ingest.py <document>"
echo "  3. Run: python3 src/query.py <corpus_id> '<question>'"
echo "========================================"
