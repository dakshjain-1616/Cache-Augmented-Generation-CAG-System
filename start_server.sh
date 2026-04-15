#!/bin/bash
# Start llama-server with optimal settings for CAG
# Uses YaRN RoPE scaling, flash attention, and KV slot persistence

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# Load configuration
if [ -f "${PROJECT_DIR}/.env" ]; then
    source "${PROJECT_DIR}/.env"
else
    echo "Error: .env file not found. Run setup.sh first."
    exit 1
fi

# Validate model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    echo "Run setup.sh to download the model."
    exit 1
fi

LLAMA_CPP_DIR="${PROJECT_DIR}/llama-cpp-turboquant"
SERVER_BIN="${LLAMA_CPP_DIR}/build/bin/llama-server"

if [ ! -f "${SERVER_BIN}" ]; then
    echo "Error: llama-server not found at ${SERVER_BIN}"
    echo "Run setup.sh to build llama.cpp."
    exit 1
fi

echo "========================================"
echo "Starting CAG llama-server"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Context Size: ${CTX_SIZE}"
echo "Port: ${LLAMA_CPP_PORT}"
echo "KV Slots Dir: ${KV_SLOTS_DIR}"
echo "GPU Available: ${GPU_AVAILABLE}"
echo "========================================"

# Calculate optimal settings
UBATCH_SIZE=128
N_PARALLEL=1

if [ "$GPU_AVAILABLE" = true ]; then
    N_GPU_LAYERS=999  # Offload all layers to GPU
    echo "Mode: GPU acceleration enabled"
else
    N_GPU_LAYERS=0
    echo "Mode: CPU only"
fi

# Start llama-server with CAG-optimized settings
# Key flags for Cache-Augmented Generation:
# --slot-save-path: Enables KV cache persistence
# --yarn: YaRN RoPE scaling for long contexts
# --flash-attn: Flash attention for efficiency
# --ubatch-size: Micro-batch size for processing

mkdir -p "${KV_SLOTS_DIR}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting server..."

exec "${SERVER_BIN}" \
    --model "${MODEL_PATH}" \
    --ctx-size "${CTX_SIZE}" \
    --port "${LLAMA_CPP_PORT}" \
    --host 0.0.0.0 \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --parallel "${N_PARALLEL}" \
    --ubatch-size "${UBATCH_SIZE}" \
    --slot-save-path "${KV_SLOTS_DIR}" \
    --rope-scaling yarn \
    --rope-scale 4.0 \
    --yarn-orig-ctx 131072 \
    --flash-attn on \
    --cache-type-k turbo3 \
    --cache-type-v turbo3 \
    --metrics \
    --slots \
    --log-colors off \
    2>&1 | tee "${PROJECT_DIR}/server.log"
