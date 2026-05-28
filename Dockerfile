# =========================================================
# CPU-ONLY PyTorch + vLLM
# AMD / Intel CPU
# =========================================================

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# =========================================================
# System packages
# =========================================================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    wget \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# Create virtual environment
# =========================================================
RUN python3 -m venv /opt/venv

# =========================================================
# Activate venv
# =========================================================
ENV PATH="/opt/venv/bin:$PATH"

# =========================================================
# Upgrade pip tooling INSIDE venv
# =========================================================
RUN pip install --upgrade \
    pip \
    setuptools \
    wheel

# =========================================================
# Install CPU-only PyTorch
# =========================================================
RUN pip install \
    torch==2.12.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# =========================================================
# Install vLLM dependencies
# =========================================================
RUN pip install \
    pyzmq \
    protobuf \
    sentencepiece \
    tiktoken \
    transformers \
    accelerate \
    gguf \
    numpy \
    scipy \
    pandas \
    requests \
    fastapi \
    uvicorn \
    ninja \
    einops \
    cachetools \
    diskcache \
    psutil \
    py-cpuinfo \
    prometheus_client

RUN pip install \
    aiohttp \
    anthropic \
    blake3 \
    cbor2 \
    compressed-tensors \
    depyf \
    fastsafetensors \
    ijson \
    lark \
    llguidance \
    lm-format-enforcer\
    mcp \
    mistral_common[image] \
    model-hosting-container-standards \
    msgspec \
    numba \
    openai \
    openai-harmony \
    opencv-python-headless \
    opentelemetry-api \
    opentelemetry-exporter-otlp \
    opentelemetry-sdk \
    opentelemetry-semantic-conventions-ai \
    outlines_core \
    partial-json-parser \
    prometheus-fastapi-instrumentator \
    pybase64 \
    python-json-logger \
    setproctitle \
    watchfiles \
    xgrammar

# =========================================================
# Install vLLM
# =========================================================
RUN pip install vllm --no-deps

# =========================================================
# Workspace
# =========================================================
WORKDIR /workspace

COPY . /workspace

CMD ["bash"]