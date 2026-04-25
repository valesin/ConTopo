# Training Loop

This document covers the design of `scripts/01_train_models.py` and
`src/training/train_ce.py`: what each component does, why it exists, and the
operational constraints that shaped those choices.

---

## 1. Loop structure

One call to `main()` trains one configuration (one rho, one topology, one
trial). Hydra MULTIRUN launches all configurations from a sweep YAML by calling
`main()` repeatedly in the **same Python process**, sequentially, using
Hydra's default `BasicLauncher`.

```
scripts/01_train_models.py
  └── main()          ← called once per MULTIRUN job
        ├── idempotency check  → skip if already FINISHED
        ├── data loaders
        ├── model + losses + balancer + optimiser
        └── for epoch in 1..N:
              train_one_epoch(...)
              validate(...)
              early stopping + best-model tracking
```

Each `main()` call is self-contained: all GPU objects (model, optimiser states,
activations) are created fresh and released at function exit. The training loop
itself (`train_one_epoch`) is stateless between calls — the only stateful object
that persists across epochs is `GradNormBalancer._lambda_hat`.

---

## 2. GradNormBalancer

### What it does

The balancer dynamically scales the topographic loss so that its gradient norm
stays at a target fraction `rho` of the task-loss gradient norm:

```
scale_t = EMA( rho × ||∇_θ L_task|| / ||∇_θ L_topo|| )
total_loss = L_task + scale_t × L_topo
```

This decouples `rho` from the raw magnitude of `L_topo` — setting `rho=0.1`
means "the topo gradient should be 10% as large as the task gradient",
regardless of what the loss values themselves are.

### Why `retain_graph=True`

`grad_norm` (in `src/losses/balancer.py`) calls `torch.autograd.grad` twice on
the same forward graph — once for `L_task` and once for `L_topo`:

```python
def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, ...)
    ...
```

`retain_graph=True` is required on the first call because the graph is shared:
both losses flow through the same backbone activations. Without it, the first
`autograd.grad` call would free the graph, making the second call fail.

The graph is finally freed by the subsequent `loss.backward()` call (which does
not use `retain_graph`), so there is no permanent accumulation within a single
training step.

### Memory cost

During `balancer.step()`, the full forward-pass computation graph is held in
GPU memory while two sets of gradients are computed. This means activation
memory is live for longer than in a vanilla training loop — roughly until
`loss.backward()` completes. For ResNet34 on ImageNet with `batch_size=64`,
this adds no measurable overhead over what `backward()` would hold anyway,
since the graph must be retained for backprop regardless.

---

## 3. CUDA OOM in MULTIRUN: root cause and fixes

### Symptom

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 100.00 MiB.
GPU 0 has a total capacity of 23.55 GiB of which 60.56 MiB is free.
...1.37 GiB is reserved by PyTorch but unallocated.
```

This occurs partway through a sweep (not on the first run), on a GPU large
enough to comfortably hold a single ResNet34 training run.

### Root cause: allocator cache accumulation across MULTIRUN jobs

PyTorch's CUDA memory allocator never returns memory to the CUDA driver
spontaneously. When a tensor is freed, its backing memory goes into a
**per-process allocator cache** and stays there for fast reuse. This is
intentional — `cudaMalloc` is expensive.

In Hydra MULTIRUN (BasicLauncher), all 60 jobs run in the same Python process.
After each `main()` call, Python GC reclaims CPU references (model, optimiser,
tensors), which frees their GPU backing memory back into the allocator cache.
But the cache itself grows with each run because:

1. Different runs allocate differently-sized tensors (different rho values,
   topology types, batch sizes). Each allocation pattern leaves differently-
   shaped holes in the cache.
2. Fragmentation: a 200 MiB free block from run N may be split into two 100 MiB
   blocks across runs N+1 and N+2. By run M, the 100 MiB allocation in
   `self.conv2(out)` cannot be satisfied from any free block, and the allocator
   cannot request more from CUDA because the GPU is (from CUDA's view) full.

The 1.37 GiB "reserved but unallocated" in the error is the fragmented cache
that cannot satisfy the new request.

### Fix 1 — `torch.cuda.empty_cache()` at the start of each job

`scripts/01_train_models.py` calls `torch.cuda.empty_cache()` (plus
`gc.collect()` to flush Python's reference cycle collector first) at the very
top of `main()`, before any GPU allocation:

```python
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

`empty_cache()` returns all cached-but-unallocated blocks to the CUDA driver.
The next run starts with a clean allocator. This is the primary fix: it breaks
the accumulation chain at each job boundary.

`gc.collect()` is called first because Python's cyclic GC may not have run
since the last `main()` returned, meaning some tensors from the previous run
may not yet have been freed from the allocator cache. Forcing a collection
ensures `empty_cache()` sees the maximum amount of reclaimable memory.

### Fix 2 — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Set in `.env.secrets`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

With `expandable_segments`, the allocator uses smaller, growable segments
instead of large fixed-size slabs. This reduces internal fragmentation — a free
region can be split and recombined more flexibly. It does not prevent cache
accumulation on its own, but it means fragmentation within each run is
lower, giving Fix 1 more room to work.

Both fixes are complementary: Fix 1 prevents inter-run accumulation; Fix 2
reduces intra-run fragmentation.

### What does NOT fix it

- **Reducing `batch_size`**: helps per-run peak memory but does not stop
  accumulation across runs. The sweep would still OOM on run N+k.
- **Enabling AMP** (`training.amp=true`): halves activation memory, giving more
  headroom, but again does not stop the accumulation mechanism.
- **Using a parallel SLURM launcher**: each job runs in its own process and
  gets its own CUDA context, so accumulation cannot occur. This is the nuclear
  option — correct, but requires SLURM configuration.

---

## 4. AMP (automatic mixed precision)

AMP is opt-in (`training.amp=false` by default). When enabled:

- The forward pass and loss computation run in `float16`.
- The `GradNormBalancer` explicitly casts both losses to `float32`
  (`task_loss.float()`, `topo_loss.float()`) before calling `autograd.grad`.
  This is required because `torch.autograd.grad` on mixed-precision graphs can
  produce `None` gradients for `float16` leaves.
- `loss.backward()` uses a `GradScaler` to prevent underflow in `float16`
  gradients.

AMP is recommended for FFCV runs (where it is part of the high-throughput
recipe) and for large-batch ImageNet training. For CIFAR-10 runs it has
negligible benefit over the small model and batch size used.

---

## 5. `cudnn.benchmark = True`

Set once per `main()` call. This tells cuDNN to profile several convolution
algorithms on the first batch and cache the fastest one for the remaining
batches. The profiling adds a few seconds to the first epoch but pays off
over a 30–200 epoch run.

Caveat: `cudnn.benchmark` is a global flag shared across all MULTIRUN jobs in
the same process. Setting it to `True` in every `main()` call is idempotent and
harmless.

---

## 6. Numba finalizer `KeyError` between MULTIRUN jobs

Between sequential MULTIRUN jobs you may see noise like:

```
Exception ignored in: <finalize object at 0x...>
  File ".../numba/core/dispatcher.py", line 268, in finalizer
    for cres in overloads.values():
KeyError: (Array(uint8, 1, 'C', True, aligned=True), ...)
```

This is a **known Numba upstream bug** (affects Python 3.12+) and is harmless.
FFCV compiles image-pipeline kernels with Numba. Numba registers weakref
finalizers to clean up those compiled functions when they are GC'd. During
inter-job GC (triggered by `gc.collect()` or naturally between calls), Python
can clear Numba's `overloads` dict *before* the finalizer fires. The finalizer
then tries to delete an already-gone key, raising `KeyError`.

The training run that preceded the message completed successfully — the error
always prints *after* `Done. test_acc=...`. The next job starts normally.
Nothing can be done from user code; the fix belongs in Numba.

---

## 7. Early stopping and best-model tracking

The training loop tracks the best model state in CPU memory
(`copy.deepcopy(model.state_dict())`), not as an intermediate MLflow artifact.
This avoids writing partial models to the artifact store and keeps the loop
simple. The best weights are loaded back into the model after the loop and
logged once as the `e2e_best` artifact.

The metric used to judge improvement is controlled by
`training.early_stopping_method`:

- `val_acc` (default): stops when validation accuracy stops improving.
- `val_loss`: stops when validation loss stops decreasing.

Both use the same patience counter (`training.early_stopping_patience`).

---

## Reference

- `scripts/01_train_models.py` — full training entrypoint
- `src/training/train_ce.py` — `train_one_epoch`, `validate`
- `src/losses/balancer.py` — `GradNormBalancer`, `grad_norm`
- `src/networks/resnet34_imagenet.py` — `FinetuneResNet34`, `ScratchResNet34`
- `conf/sweeps/imagenet100_t5.yaml` — torch-backend ImageNet100 sweep (30 ep, bs=64)
- `conf/sweeps/imagenet100_ffcv_t5.yaml` — FFCV sweep (16 ep, bs=1024, AMP-ready)
- `docs/ffcv_param_assumptions.md` — FFCV recipe parameter migration notes
