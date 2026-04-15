# Cache-Augmented Generation (CAG) System Dockerfile
# Multi-stage build for efficient deployment

FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone and build llama.cpp with CUDA support
RUN git clone https://github.com/atomicmilkshake/llama-cpp-turboquant llama-cpp-turboquant && \
    cd llama-cpp-turboquant && \
    mkdir -p build && cd build && \
    cmake .. \
        -DLLAMA_BUILD_SERVER=ON \
        -DGGML_CUDA=ON \
        -DLLAMA_FLASH_ATTN=ON \
        -DCMAKE_CUDA_ARCHITECTURES=all && \
    make -j$(nproc)

# Production stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy llama.cpp binaries from builder
COPY --from=builder /app/llama-cpp-turboquant/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /app/llama-cpp-turboquant/build/bin/llama-cli /usr/local/bin/llama-cli

# Copy application files
COPY requirements.txt .
COPY setup.sh .
COPY start_server.sh .
RUN mkdir -p /app/src
COPY src/ingest.py src/ingest.py
COPY src/query.py src/query.py
COPY src/api_server.py src/api_server.py
COPY src/demo.py src/demo.py

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/models /app/kv_slots /app/corpora

# NOTE: Do not bake the model into the image — mount ./models as a volume
# Download the model before running: see README.md for instructions

# Create initial manifest
RUN echo '{"corpora": [], "version": "1.0"}' > /app/manifest.json

# Create .env file with defaults
RUN echo 'GPU_AVAILABLE=true' > /app/.env && \
    echo 'GPU_VRAM_MB=8192' >> /app/.env && \
    echo 'MODEL_PATH=/app/models/qwen2.5-3b-instruct-q4_k_m.gguf' >> /app/.env && \
    echo 'MODEL_NAME=qwen2.5-3b-instruct-q4_k_m.gguf' >> /app/.env && \
    echo 'CTX_SIZE=8192' >> /app/.env && \
    echo 'LLAMA_CPP_PORT=8080' >> /app/.env && \
    echo 'API_PORT=8000' >> /app/.env && \
    echo 'KV_SLOTS_DIR=/app/kv_slots' >> /app/.env && \
    echo 'CORPORA_DIR=/app/corpora' >> /app/.env

# Make scripts executable
RUN chmod +x /app/setup.sh /app/start_server.sh

# Expose ports
# 8080: llama-server
# 8000: CAG API server
EXPOSE 8080 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: start llama-server
# Override with docker-compose for multi-service setup
CMD ["./start_server.sh"]
