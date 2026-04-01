FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace/ConTopo

# Install dependencies (cached layer — only busted when training deps actually change)
# requirements-training.txt is generated from pyproject.toml — see README
COPY requirements-training.txt ./
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    -r requirements-training.txt

# Copy full project (code + dataset)
COPY . .

# Download CIFAR-10 if not present in build context
RUN if [ ! -d "dataset/cifar-10-batches-py" ]; then \
    python -c "import torchvision; torchvision.datasets.CIFAR10('dataset', download=True)"; \
    fi

ENV PYTHONPATH=/workspace/ConTopo

CMD ["bash"]
