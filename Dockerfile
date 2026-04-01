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
RUN uv pip install --system --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    -r requirements-training.txt

# Copy dataset as a separate layer so updates to code won't invalidate it.
# If you prefer to download during build instead, keep the fallback below.
COPY dataset/ dataset/

# Download CIFAR-10 if dataset was not provided in build context
RUN if [ ! -d "dataset/cifar-10-batches-py" ]; then \
    python -c "import torchvision; torchvision.datasets.CIFAR10('dataset', download=True)"; \
    fi

# Copy rest of project (code) — `.dockerignore` excludes dataset/ so this
# COPY won't include the dataset layer again and won't bust the dataset layer
# when code changes.
COPY . .

ENV PYTHONPATH=/workspace/ConTopo

CMD ["bash"]
