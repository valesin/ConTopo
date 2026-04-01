FROM mambaorg/micromamba:1-ubuntu22.04

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

# Install full conda environment from existing spec
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Activate base env for subsequent RUN steps
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy project (code + dataset, no .git or secrets)
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace/ConTopo
WORKDIR /workspace/ConTopo
ENV PYTHONPATH=/workspace/ConTopo

# Entrypoint: micromamba activation → secrets sourcing → user command
COPY --chown=$MAMBA_USER:$MAMBA_USER docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
