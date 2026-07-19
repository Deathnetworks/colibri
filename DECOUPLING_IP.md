# DECOUPLING_IP.md — Decoupled Attention (GPU) / FFN (CPU) Integration Spec

> Supersedes `DECOUPLING_IP.old.md`. This document is a concrete implementation
> progression for folding the best elements of `llama.cpp-PoC` into `colibrì`,
> raising decode throughput from ~0.07 tok/s (the current cold-expert disk-bound
> floor on VRAM-constrained hardware) toward 5+ tok/s by maximising the use of
> BOTH the GPU (attention + KV) and the CPU (FFN matmul) at the same time, while
> hiding SSD latency behind GPU compute via DirectStorage-class async I/O and
> ReBAR/zero-copy residual transfer.

---

## 0. Situation Report (what each repo actually does today)

### colibrì (`c/glm.c` + `c/backend_cuda.cu` / `backend_sycl.cpp`)
colibrì is a **single-file, layer-explicit MoE engine**. It already does a
*partial* version of what we want:

- **Attention runs on the GPU.** `attention_rows()` (glm.c:2428) drives
  `coli_cuda_attention_absorb_batch_dev` / `coli_cuda_attention_project_batch_dev`
  on a CUDA (or SYCL) stream. KV cache lives on device.
- **FFN (the MoE experts) runs on the CPU.** `moe()` (glm.c:2757) does routing,
  loads expert weights from disk, and runs `matmul_i4` / `matmul_q` (AVX2/AVX512
  / NEON dequant dot products) on the CPU.
- **The two are already overlapped.** In the layer decode path
  (`layer_cuda_shard_kvb` + the `coli_cuda_pipe_*` calls around glm.c:3690-3727),
  the GPU attention for layer N is launched asynchronously while the CPU is
  *already inside* `moe()` for the same layer, and the residual add is posted
  back to the device with `coli_cuda_pipe_add` (async).
- **Async disk I/O already exists.** `uring.h` is a hand-rolled `io_uring`
  wrapper (`coli_uring_prep_read` / `coli_uring_enter` / `coli_uring_peek`),
  and `compat.h` provides `compat_pread` (ReadFile + OVERLAPPED on Windows,
  `O_DIRECT`-twin on Linux). The expert loader (`expert_load`, `uring_load_add`,
  `pipe_dispatch`) uses these to stream expert weights layer-by-layer.
- **Quant format is known.** `QT` weights are INT4-packed (2 nibbles/byte) with
  per-group scales; `matmul_i4_grouped` already dequantizes on the fly with
  AVX2/AVX512/NEON. No separate F32 buffer needed.

So colibrì is **~70% of the way there**. What it is missing versus the PoC's
vision is *explicit, correct decoupling primitives*: zero-copy residual transfer,
true N+1 prefetch overlap, and a clean "FFN weights on CPU/RAM, attention on GPU"
contract that is currently implicit and partially serialised by `cudaMemcpyAsync`
and synchronous expert waits.

### llama.cpp-PoC (`src/llama-ffn-*.{cpp,h}`, `docs/IMPLEMENTATION_PLAN.md`)
The PoC is a **ggml-graph-based** fork. Its decoupling subsystem is the prize:

- **`ffn_mode_t`** enum: `FFN_GPU`, `FFN_LOCAL` (FFN in CPU RAM, attention on GPU),
  `FFN_ZERO_CPU` (FFN mmap'd from SSD, attention on GPU), `FFN_BUFFERED_GPU`.
- **`ffn_async_buffer`** double-buffer engine: two 4 KB-aligned buffers,
  `active_idx`/`prefetch_idx` ping-pong, `ffn_async_init/load_layer/free`.
- **`ffn_layer_ptrs_t`**: per-layer file offsets for `ffn_norm/gate/up/down`,
  with `async_valid` flag — i.e. the model loader records byte offsets so a
  whole FFN layer can be `pread()`'d contiguously.
- **`llm_compute_ffn_cpu`**: CPU FFN compute with **on-the-fly Q4→F32 dequant**
  via `ggml_get_type_traits(type)->to_float` (row-by-row), RMSNorm + SiLU + gate/up/down.
- **DirectStorage intent**: `O_DIRECT` (Linux) / `FILE_FLAG_NO_BUFFERING` (Win)
  opening, falling back gracefully; `aligned_alloc_4k` for 4096-byte alignment.
- **Architecture diagram** (IMPLEMENTATION_PLAN.md): per-token flow = GPU attention
  N → CPU reads FFN N+1 → GPU→CPU residual copy (4 KB) → CPU dequant+FFN →
  CPU→GPU result copy (4 KB) → evict layer N. The key wins: only 1 layer's FFN
  weights in RAM at a time, and disk I/O for layer N+1 is hidden behind GPU
  attention for layer N.

**Critical caveat about the PoC:** its build is broken (CODE_MAP.md "Build
currently fails"), `ffn_async_load_layer` still does `memcpy` from mmap rather
than `O_DIRECT pread`, and io_uring/IOCP are stubbed. So we take the PoC's
*design and data structures*, NOT its code. colibrì's existing `uring.h` +
`compat.h` are already more complete than the PoC's stubs.

---

## 1. Overlap Analysis (where the logic coincides)

| Concern | colibrì today | PoC | Best element to keep |
|---|---|---|---|
| Attention | GPU (`backend_cuda.cu`/`backend_sycl.cpp`) | GPU via ggml backend | colibrì's explicit kernel path (already works, keep) |
| FFN compute | CPU (`moe()` + AVX matmul) | CPU (`llm_compute_ffn_cpu`) | Both — colibrì's AVX2/AVX512/NEON dequant matmul is faster/more portable than PoC's naive loop; keep colibrì, adopt PoC's *structure* |
| Weight storage | `QT` int4 + per-group scale, streamed per expert | `ffn_layer_ptrs_t` file offsets, contiguous per-layer read | Adopt PoC's **per-layer contiguous offset map** for the dense/shared FFN and the *routed expert batch*, so one `pread` fetches a whole layer |
| Disk I/O | `uring.h` io_uring + `compat_pread` OVERLAPPED, expert-streamed | stubs (`read()`/`memcpy`) | colibrì's real async I/O — extend to **double-buffered ping-pong** |
| Residual transfer | `cudaMemcpyAsync` H⇄D for ~S·D floats | `memcpy` via ReBAR mapped memory | PoC's **ReBAR/zero-copy** idea — map VRAM into CPU addr space, replace small `cudaMemcpyAsync` with `memcpy` |
| Overlap scheduling | implicit (GPU async + CPU moe) | explicit N+1 prefetch in layer loop | adopt PoC's **explicit prefetch of layer N+1 before computing layer N** |
| RAM budget | bounded by expert LRU/pin cache | 1 layer FFN in RAM | keep colibrì's LRU; the decoupling makes the *cold* path (no RAM for experts) viable at 5+ t/s |

**Conclusion:** colibrì is the base. We scavenge from the PoC: (a) the
ReBAR/zero-copy residual primitive, (b) the explicit N+1 prefetch + double-buffer
contract, (c) the per-layer contiguous offset bookkeeping so a whole FFN block is
one aligned `pread`. We do NOT port ggml, the PoC build, or its naive matmul.

---

## 2. Target Architecture (the contract)

```
┌─────────────┐         mapped VRAM (ReBAR)          ┌──────────────┐
│   GPU       │  ◄──── residual stream (S×D f32) ────│   CPU        │
│ Attention   │       written by CPU via memcpy      │ FFN matmul  │
│ + KV cache  │       read by GPU immediately        │ (AVX matmul)│
│ (async)     │                                      │ + dequant    │
└──────┬──────┘                                      └──────┬───────┘
       │                                                       │
       │  coli_cuda_attention_* (async stream)                 │  reads FFN weights
       │                                                       │  from double buffer
       ▼                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  NVMe SSD  ──O_DIRECT/io_uring pread──►  ffn_async_buffer[2] (RAM)     │
│  FFN weights (int4, per-layer contiguous block)                       │
└─────────────────────────────────────────────────────────────────────┘

Per-token / per-layer decode loop (layer N):
  1. (before loop, layer 0) synchronously load FFN for layer 0.
  2. Issue async prefetch of FFN weights for layer N+1  ──► hides disk behind GPU.
  3. Launch GPU attention for layer N on the CUDA/SYCL stream (async).
  4. Wait for layer N's FFN prefetch (issued in step 2 of iteration N-1).
  5. cudaStreamSynchronize(attention) — GPU attention done.
  6. CPU FFN: dequant int4→f32 on the fly, gate/up → SiLU → down, write result
     directly into the ReBAR-mapped residual buffer (no cudaMemcpy).
  7. Next layer. The residual add to VRAM is a memcpy the GPU sees immediately.
```

This is exactly the PoC's pipeline, but expressed against colibrì's existing
`coli_cuda_pipe_*` / `moe()` machinery instead of ggml.

---

## 3. Implementation Progression (phased, each independently testable)

Each phase is a self-contained, buildable, measurable step. Do them in order;
every phase must show a throughput/ latency win (or a documented no-regression)
before the next begins. Target measurement: `make -C c bench` + a fixed
prompt at a fixed `--ram`/`--ngen`, on the reference VRAM-constrained host.

### Phase 0 — Instrumentation & Baseline (prerequisite)
- Add a `DECOUPLE=0|1` global and a `decouple_prof` struct counting: SSD service
  time for FFN loads, GPU attention busy time, CPU FFN busy time, residual
  copy time, and the "stall" time where CPU waits on disk or GPU waits on CPU.
- Establish the **0.07 t/s baseline** number and a flame of where time goes
  (likely: CPU blocked in synchronous expert `pread`, or GPU idle waiting on
  `cudaMemcpyAsync` of the residual).
- Files: `glm.c` (globals + counters), `c/coli` (no change), `docs/`.

### Phase 1 — Per-layer contiguous FFN offset map (the bookkeeping the PoC proved necessary)
- At model load, record for each layer the **byte offsets + sizes** of the FFN
  weight block (shared expert `gate/up/down` + each routed expert's
  `gate/up/down`, or, for dense layers, the single MLP). Store in a
  `ffn_layer_ptrs_t`-equivalent (`Layer` extension or a parallel array).
- Goal: one aligned `pread()` fetches a whole layer's FFN (or a whole expert
  group) instead of many small seeks. Reuses colibrì's existing `st.h` tensor
  offset data — just surface it.
- Adopt from PoC: `ffn_layer_ptrs_t` shape, `async_valid` gating.
- Files: `glm.c` (load path), `st.h`/`tier.h` (struct), no backend change.

### Phase 2 — ReBAR / Zero-Copy residual stream (kills the `cudaMemcpyAsync` tax)
- Add `coli_cuda_alloc_mapped(bytes, &device_ptr)` → `cudaHostAllocMapped` +
  `cudaHostGetDevicePointer` (CUDA), and the SYCL/Vulkan equivalents
  (`sycl::malloc_host` / `cl::sycl::usm::allocator` shared, or `vkMapMemory`
  mapped staging). This is the function stubbed in `backend_cuda.h` already
  (`coli_cuda_alloc_mapped` is declared).
- Route the residual stream buffers (`x_dev`/host mirror, `out`, the per-layer
  `nrm`/`hc` working buffers used in the `coli_cuda_pipe_*` path) through mapped
  memory.
- Replace the small `cudaMemcpyAsync` H⇄D residual transfers with plain `memcpy`
  into the mapped buffer (GPU sees it via PCIe/ReBAR immediately).
- Robust fallback: if `cudaHostAllocMapped` fails or ReBAR is off, fall back to
  the current `cudaMemcpyAsync` path (feature flag `REBAR=0` disables).
- Reuse: the `engine_alloc_mapped` / `engine_free_mapped` shims already added to
  glm.c (grep `engine_alloc_mapped`) — they just need the backend to actually
  back them. **This is the single highest-leverage change.**
- Files: `backend_cuda.cu` (implement `coli_cuda_alloc_mapped`/`free_mapped`),
  `backend_sycl.cpp`, `backend_vulkan.cpp` (mapped equivalents), `glm.c`
  (use `engine_alloc_mapped` for residual buffers).

### Phase 3 — Double-buffered async FFN I/O (true DirectStorage overlap)
- Build `ffn_async_buffer` (mirror PoC struct) on top of colibrì's **existing**
  `uring.h` (Linux) and `compat_pread` OVERLAPPED (Windows) — do NOT use the
  PoC's stubbed `read()`/`memcpy`.
- `ffn_async_init`: open model fd with `O_DIRECT` (Linux, via `compat_open_direct`)
  / `FILE_FLAG_NO_BUFFERING` (Win), allocate two 4 KB-aligned buffers sized to
  max FFN layer (reuse `aligned_alloc_4k` pattern from PoC).
- `ffn_async_prefetch(layer N+1)`: submit an `io_uring` SQE (or OVERLAPPED read)
  for the whole layer block; returns immediately.
- `ffn_async_swap()`: `io_uring_enter(min_complete=1)` / `GetOverlappedResult`
  waits for the in-flight read, then ping-pongs `active_idx`/`prefetch_idx`.
- Wire into `moe()` / the layer decode loop: prefetch N+1 at step 2, consume N
  at step 4. Replaces the current synchronous `expert_load` wait with an
  already-in-flight read.
- Files: new `c/ffn_async.h` + `c/ffn_async.c` (OS-abstracted, uses `uring.h` +
  `compat.h`), `glm.c` (call sites), `Makefile` (build `ffn_async.c`).

### Phase 4 — Explicit decoupled layer pipeline (the overlap schedule)
- Restructure the per-layer decode function (the block around glm.c:3690-3727)
  to the 7-step contract in §2: prefetch N+1 → GPU attention N (async) → wait
  FFN N → sync GPU → CPU FFN N (into mapped residual) → repeat.
- Ensure the GPU attention stream and the CPU FFN never serialise on a copy:
  the residual is in mapped VRAM (Phase 2), so step 6's write is visible to the
  GPU at step 1 of layer N+1 with zero driver call.
- Keep colibrì's existing `g_pipe`/`PILOT_REAL` machinery as the *fallback*
  schedule when `DECOUPLE=0` or on systems without `O_DIRECT`/io_uring.
- Files: `glm.c` (layer loop), `backend_cuda.cu` (already async — verify stream
  ordering), `backend_sycl.cpp` (same for SYCL queue).

### Phase 5 — On-the-fly dequant + CPU FFN kernel hardening (latency, not correctness)
- Port PoC's `llm_compute_ffn_cpu` *structure* but keep colibrì's AVX2/AVX512/NEON
  `matmul_i4_grouped` as the inner kernel (it is already vectorised and faster
  than the PoC's scalar loop). Specifically: ensure the FFN path dequantizes
  row-by-row into the registers / L1 rather than materialising a full F32 layer
  buffer (RAM budget: only int4 weights + small f32 accum in flight).
- Add OpenMP parallelism to the CPU FFN (colibrì already uses `#pragma omp` in
  `matmul_i4`) so the FFN saturates all CPU cores while the GPU does attention.
- Files: `glm.c` (`moe` / `dense_mlp`), possibly `c/ffn_cpu.h` if extracted.

### Phase 6 — Windows + SYCL/Vulkan parity & robustness
- SYCL backend: mapped host memory via `sycl::malloc_host` (USM shared) — the
  `coli_sycl_alloc_mapped` stub already exists; implement it.
- Vulkan backend: `vkMapMemory` a staging buffer; `coli_vulkan_alloc_mapped`.
- Windows: `FILE_FLAG_NO_BUFFERING` + IOCP/Overlapped (colibrì's `compat_pread`
  already does OVERLAPPED; extend to the double-buffer).
- Fallbacks: no `O_DIRECT` → buffered `pread`; no io_uring (old kernel) →
  thread-pool `pread` + `pipe`; no ReBAR → `cudaMemcpyAsync`. Every fallback
  must keep the pipeline correct (just slower).

### Phase 7 — 1T-class validation & tuning
- Run the reference >100B model; confirm RAM stays bounded (≤1-2 layer FFN in
  RAM) and throughput ≥ 5 t/s on the VRAM-constrained host.
- Tune `URING` workers, prefetch depth (consider N+2 lookahead), and the
  ReBAR residual buffer sizing. Add a `DECOUPLE_BUDGET` knob for max RAM.

---

## 4. Reuse Map (colibrì assets we keep, PoC assets we scavenge)

| Asset | Source | Use |
|---|---|---|
| `uring.h` (`coli_uring_*`) | **colibrì** | async I/O engine for Phase 3 (already real, not stubbed) |
| `compat.h` `compat_pread` / `compat_open_direct` | **colibrì** | cross-OS O_DIRECT / OVERLAPPED |
| `matmul_i4` / `matmul_i4_grouped` (AVX/NEON) | **colibrì** | CPU FFN inner kernel (Phase 5) |
| `coli_cuda_pipe_*` (async residual add/upload) | **colibrì** | the async GPU attention+residual path |
| `engine_alloc_mapped` / `engine_free_mapped` shims | **colibrì** | hook point for ReBAR (Phase 2) |
| `attention_rows` / `coli_cuda_attention_*` | **colibrì** | GPU attention (keep as-is) |
| `ffn_async_buffer` / `ffn_layer_ptrs_t` struct | **PoC** | double-buffer + offset-map shape (Phase 1/3) |
| `llm_compute_ffn_cpu` dequant structure | **PoC** | CPU FFN *structure* (Phase 5, with colibrì kernel) |
| `ffn_mode_t` (FFN_LOCAL / ZERO_CPU) | **PoC** | mode flags → colibrì `DECOUPLE` / `REBAR` globals |
| ReBAR/zero-copy residual concept | **PoC** | Phase 2 design (implemented via colibrì backends) |

---

## 5. Risks & Mitigations (from both repos' known issues)

- **CPU becomes the new bottleneck (PoC doc, §3).** Mitigation: OpenMP-parallel
  AVX-512 FFN (colibrì already has it) + keep attention fully on GPU so CPU is
  free for FFN only. Measure CPU FFN time vs GPU attention time; if CPU-bound,
  raise `DECOUPLE_BUDGET` to pin hot experts in RAM (colibrì's existing LRU).
- **Dequant overhead.** Mitigation: row-by-row dequant into registers (Phase 5),
  never a full F32 layer buffer.
- **O_DIRECT 4 KB alignment.** Mitigation: reuse PoC's `aligned_alloc_4k`; assert
  buffer/offset alignment; fall back to buffered `pread` if `O_DIRECT` rejected.
- **ReBAR unavailable / mapped-alloc fails.** Mitigation: `REBAR=0` → current
  `cudaMemcpyAsync` path; feature-detect at init.
- **Build breakage (PoC's own failure mode).** Mitigation: we never adopt the
  PoC build; we extend colibrì's working `Makefile` + `backend_*` DLL model.
- **Cross-backend drift (CUDA/SYCL/Vulkan).** Mitigation: the mapped-alloc and
  async-I/O APIs are already declared in all three backend headers — implement
  each, gate by `COLI_CUDA`/`COLI_SYCL`/`COLI_VULKAN`.

---

## 6. Acceptance Criteria

1. `DECOUPLE=1 REBAR=1 URING=1` on the VRAM-constrained reference host yields
   **≥ 5 tok/s** decode (from the ~0.07 t/s cold floor), with RAM usage bounded
   to ≤ ~2 layers of FFN weights resident.
2. The GPU attention stream and CPU FFN matmul are **concurrently busy** for the
   majority of each layer (proven by `decouple_prof`: disk + GPU + CPU overlap,
   not serial waits).
3. All three backends (CUDA, SYCL, Vulkan) build and run the decoupled path;
   each gracefully falls back when ReBAR / O_DIRECT / io_uring unavailable.
4. `make -C c check` and the existing C/Python test suites pass unchanged (the
   decoupling is a runtime scheduling change, not a numerics change — outputs
   are bit-equivalent to the serial path modulo floating-point reduction order).
5. No regression when `DECOUPLE=0` (existing pipeline preserved as fallback).

---

## 7. Suggested PR / branch strategy
- Branch `decoupled-split` off `main`.
- One PR per phase (0→7), each with a benchmark delta in the PR body.
- Keep `DECOUPLE` / `REBAR` / `URING` defaulting to the existing behaviour until
  Phase 4 proves the win, then flip defaults in a final PR.

---

*Source material:* `colibrì` `c/glm.c`, `c/backend_cuda.cu`, `c/uring.h`,
`c/compat.h`, `DECOUPLING_IP.old.md`; `llama.cpp-PoC` `src/llama-ffn-async.cpp`,
`src/llama-ffn-local.{h,cpp}`, `docs/IMPLEMENTATION_PLAN.md`, `docs/CODE_MAP.md`,
`docs/SESSION_CONTINUITY_FINAL.md`.
