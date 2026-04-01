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

# Dataset root — where CIFAR-10 is downloaded at build time and read at runtime.
# Override at build time:  docker build --build-arg CONTOPO_DATA_ROOT=mydata .
# Override at runtime:     docker run -e CONTOPO_DATA_ROOT=/data -v /host/data:/data ...
ARG CONTOPO_DATA_ROOT=./dataset
ENV CONTOPO_DATA_ROOT=${CONTOPO_DATA_ROOT}

# Download CIFAR-10 into its own layer. Docker caches this layer by
# (command text + previous layer hash), so it only re-runs if dependencies
# change. Once pushed to the registry, any machine pulls the cached layer.
RUN python -c "import torchvision; torchvision.datasets.CIFAR10('${CONTOPO_DATA_ROOT}', download=True)"

# Code in a separate layer — changes here don't re-push the dataset
COPY src/ src/
COPY scripts/ scripts/
COPY conf/ conf/
COPY tests/ tests/
COPY main.py __init__.py pyproject.toml ./

ENV PYTHONPATH=/workspace/ConTopo

CMD ["bash"]
