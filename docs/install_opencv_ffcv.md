# Install OpenCV with micromamba (no sudo)

This guide shows how to install **OpenCV** in user space using **micromamba**, without requiring `sudo`. It also includes a “native-deps-only” workflow where you keep using your existing Python environment (e.g., `uv` / `venv`) while using micromamba only to supply the **OpenCV C/C++ libraries + `opencv4.pc`** needed by build tools like `pkg-config`.

> Tested workflow idea: create a micromamba env named `opencv-only`, then expose its `pkg-config` metadata and shared libraries via environment variables.

---

## Prerequisites

- You have `micromamba` available on the machine (installed in your `$HOME` or provided by the system).
- You have basic build tools already available (compiler, etc.) if you’re building packages that compile native extensions.
- You do **not** need `sudo`.

---

## 1) Create a micromamba environment with OpenCV

Create an environment (example name: `opencv-only`) and install OpenCV plus `pkg-config`:

```bash
micromamba create -n opencv-only -c conda-forge opencv pkg-config
```

This installs:
- OpenCV libraries and headers (as packaged by conda-forge)
- `pkg-config`, which helps other software discover OpenCV via `opencv4.pc`

---

## 2) If `micromamba activate` doesn’t work (common on HPC)

On some systems, `micromamba activate` cannot modify your current shell until you initialize the shell hook.

### Option A: Use `micromamba run` (recommended on HPC)

This does not require shell initialization:

```bash
micromamba run -n opencv-only pkg-config --modversion opencv4
```

### Option B: Initialize your shell so you can `activate`

For `bash`:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate opencv-only
```

Then test:

```bash
pkg-config --modversion opencv4
```

---

## 3) Verify OpenCV is installed (and discoverable)

### Confirm `opencv4.pc` exists

```bash
micromamba run -n opencv-only bash -lc 'find "$CONDA_PREFIX" -name opencv4.pc -o -name opencv.pc | head -n 20'
```

Typical location:

- `<env_prefix>/lib/pkgconfig/opencv4.pc`

### Confirm `pkg-config` can resolve OpenCV

```bash
micromamba run -n opencv-only bash -lc 'pkg-config --modversion opencv4'
micromamba run -n opencv-only bash -lc 'pkg-config --cflags --libs opencv4'
```

If `opencv4` is not found, check whether the pc file is named `opencv.pc` instead and try:

```bash
micromamba run -n opencv-only bash -lc 'pkg-config --modversion opencv || true'
```

---

## 4) Use micromamba OpenCV from *another* Python environment (native-deps-only)

If you want micromamba **only** for native dependencies (OpenCV + `pkg-config`) while keeping Python packages in a different environment (e.g., `uv`), export these variables in the shell where you build/run:

### 4.1 Set the environment prefix

Replace the path below with your actual env location. You can find it with:

```bash
micromamba env list
```

Then set:

```bash
export MAMBA_ENV_PREFIX="$HOME/micromamba/envs/opencv-only"
```

### 4.2 Export `PKG_CONFIG_PATH` (build-time discovery)

This is what makes `pkg-config --modversion opencv4` work outside micromamba:

```bash
export PKG_CONFIG_PATH="$MAMBA_ENV_PREFIX/lib/pkgconfig:$MAMBA_ENV_PREFIX/share/pkgconfig:$PKG_CONFIG_PATH"
```

Verify:

```bash
pkg-config --modversion opencv4
```

### 4.3 Export `LD_LIBRARY_PATH` (runtime library loading)

This helps the dynamic linker find OpenCV shared libs (`libopencv*.so`) at runtime:

```bash
export LD_LIBRARY_PATH="$MAMBA_ENV_PREFIX/lib:$LD_LIBRARY_PATH"
```

### 4.4 (Optional) Put micromamba tools on `PATH`

This ensures the micromamba-provided `pkg-config` is found:

```bash
export PATH="$MAMBA_ENV_PREFIX/bin:$PATH"
```

---

## 5) Example: building a package that requires OpenCV via pkg-config

Some packages (e.g., `ffcv`) run `pkg-config opencv4` during build. With the exports from section 4 in place:

```bash
# Example using uv:
uv pip install --no-cache-dir ffcv
```

If you already tried and it cached a failed build, force a rebuild:

```bash
uv pip install -v --no-cache-dir ffcv
```

---

## 6) Make it repeatable (recommended)

The project ships `scripts/activate-ffcv-env.sh` for exactly this purpose. Source it
before building or running anything that needs OpenCV:

```bash
source scripts/activate-ffcv-env.sh
pkg-config --modversion opencv4
```

If your micromamba is installed in a non-default location, override the prefix:

```bash
MAMBA_ENV_PREFIX=/custom/path source scripts/activate-ffcv-env.sh
```

Re-source in every new shell session. The script sets `PKG_CONFIG_PATH`,
`LD_LIBRARY_PATH`, and `PATH` — it does not activate the micromamba environment,
so uv's Python interpreter is never shadowed.

---

## Troubleshooting

### “Package opencv4 was not found in the pkg-config search path”
- Confirm the pc file exists:

  ```bash
  micromamba run -n opencv-only bash -lc 'find "$CONDA_PREFIX" -name opencv4.pc -o -name opencv.pc | head'
  ```

- Export the directory containing `opencv4.pc` into `PKG_CONFIG_PATH`:

  ```bash
  export PKG_CONFIG_PATH="$MAMBA_ENV_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
  ```

### Runtime error: `libopencv_*.so: cannot open shared object file`
Add the OpenCV lib directory to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$MAMBA_ENV_PREFIX/lib:$LD_LIBRARY_PATH"
```

### `micromamba activate` fails with “Shell not initialized”
Use `micromamba run` (no shell modification needed), or initialize your shell:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate opencv-only
```

---

## Uninstall / cleanup

Remove the environment:

```bash
micromamba env remove -n opencv-only
```

List remaining environments:

```bash
micromamba env list
```