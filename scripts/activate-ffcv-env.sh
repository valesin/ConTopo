#!/usr/bin/env bash
# Source this script before running `uv sync --group ffcv` or any ffcv training command.
# It exposes micromamba's OpenCV native libraries to uv's Python venv without
# activating the micromamba environment (which would conflict with uv's Python).
#
# Usage:
#   source scripts/activate-ffcv-env.sh
#
# Override the prefix if your micromamba is installed in a non-default location:
#   MAMBA_ENV_PREFIX=/custom/path source scripts/activate-ffcv-env.sh
#
# See docs/install_opencv_ffcv.md for the one-time micromamba setup.

MAMBA_ENV_PREFIX="${MAMBA_ENV_PREFIX:-$HOME/micromamba/envs/opencv-only}"
export PKG_CONFIG_PATH="$MAMBA_ENV_PREFIX/lib/pkgconfig:$MAMBA_ENV_PREFIX/share/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$MAMBA_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PATH="$MAMBA_ENV_PREFIX/bin:$PATH"
# Force micromamba's libjpeg to be loaded first. Without this, another Python
# extension (e.g. Pillow) may load the system libjpeg.so.8 before ffcv imports,
# and the dynamic linker will reuse that cached copy when libtiff.so.6 needs it,
# causing "undefined symbol: jpeg12_write_raw_data" at ffcv import time.
export LD_PRELOAD="$MAMBA_ENV_PREFIX/lib/libjpeg.so.8${LD_PRELOAD:+:$LD_PRELOAD}"
