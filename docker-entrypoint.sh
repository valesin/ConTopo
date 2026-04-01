#!/bin/bash
set -e
if [ -f /workspace/ConTopo/.env.secrets ]; then
    source /workspace/ConTopo/.env.secrets
fi
exec "$@"
