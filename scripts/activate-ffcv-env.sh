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
# Preload micromamba's image codec libs before any other Python extension can
# cache the system versions. Without this, extensions like Pillow load the system
# libjpeg/libpng early; when ffcv then imports libopencv_imgcodecs, the dynamic
# linker reuses the cached system libs (which are older and missing symbols like
# jpeg12_write_raw_data or png_set_cICP), ignoring micromamba's newer copies.
_ffcv_preload=""
for _lib in libjpeg.so.8 libpng16.so.16 libtiff.so.6 libwebp.so.7; do
    [ -f "$MAMBA_ENV_PREFIX/lib/$_lib" ] && \
        _ffcv_preload="$MAMBA_ENV_PREFIX/lib/$_lib${_ffcv_preload:+:$_ffcv_preload}"
done
export LD_PRELOAD="${_ffcv_preload}${LD_PRELOAD:+:$LD_PRELOAD}"
unset _lib _ffcv_preload
