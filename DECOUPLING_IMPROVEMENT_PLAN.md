# Decoupled Attention & FFN Improvement Plan

This document outlines the improvements that can be extracted from the `llama.cpp-PoC` repository and integrated into the current `colibrì` codebase to significantly improve speed and throughput, specifically targeting VRAM-limited consumer hardware.

## 1. Summary of Improvements

The `llama.cpp-PoC` demonstrates a highly optimized approach to running massive models on consumer hardware by separating the execution of Attention and FFN (Feed-Forward Network) blocks. The core concepts are:

1. **Decoupled Attention (GPU) and FFN (CPU)**:
   Attention blocks and the KV cache remain pinned in GPU VRAM (where their heavy memory-bandwidth requirements are met). FFN weights, which represent the bulk of the model's parameters, are streamed from an NVMe SSD to CPU RAM layer-by-layer, where the FFN compute occurs.
2. **True Async I/O & DirectStorage**:
   Bypassing the OS page cache entirely using `O_DIRECT` (Linux) and `FILE_FLAG_NO_BUFFERING` (Windows). Crucially, the PoC uses true asynchronous I/O (`io_uring` on Linux, IOCP Overlapped I/O on Windows) into a double-buffer, overlapping the disk read of layer `N+1` with the GPU attention compute of layer `N`.
3. **ReBAR (Resizable BAR) / Zero-Copy Transfers**:
   Instead of using driver-level DMA copies (like `cudaMemcpy`) for small residual stream transfers between the GPU and CPU, the PoC maps GPU VRAM directly into the CPU's address space. The CPU uses a simple `memcpy` to write the FFN output directly into VRAM, eliminating driver overhead and latency for small payloads.

Integrating these three paradigms into `colibrì` will yield lower latency per token, hide SSD read latency behind GPU compute, and eliminate CPU-GPU synchronization bottlenecks.

---

## 2. Specification & Step-by-Step Improvement Plan

### Goal
Implement ReBAR-mapped residual streams, true async I/O double-buffering, and overlapped Decoupled Attention/FFN execution in `colibrì`.

### Step 1: ReBAR-Mapped Residual Stream Buffers (Zero-Copy)

**Why it's superior**:
Currently, `colibrì` uses `cudaMemcpyAsync` to transfer the residual stream (`q`, `latent`, `rope`, `out`) between CPU and GPU. For small payloads (4KB - 16KB), the PCIe latency and CUDA driver overhead of launching a DMA operation dominate. By utilizing Resizable BAR, we map VRAM to the CPU. The CPU can write directly to the GPU via the PCIe bus (`memcpy`), which is vastly faster for small tensors.

**Implementation (Pseudo-code)**:
```c
// 1. In c/backend_cuda.cu: Expose mapped memory allocation
extern "C" void* coli_cuda_alloc_mapped(size_t bytes, void** device_ptr) {
    void* host_ptr;
    // Allocate zero-copy pinned memory (or ReBAR mapped device memory)
    cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(device_ptr, host_ptr, 0);
    return host_ptr;
}

// 2. In c/glm.c: Use mapped memory for the residual stream (h_buf)
void* d_residual;
float* h_residual = coli_cuda_alloc_mapped(S * D * sizeof(float), &d_residual);

// 3. To transfer CPU -> GPU:
// Instead of cudaMemcpyAsync(d_residual, h_residual, ...);
memcpy(h_residual, cpu_ffn_output, size);
// CPU writes directly to VRAM via ReBAR, visible to GPU immediately.
```

### Step 2: Double-Buffered True Async I/O (DirectStorage & io_uring)

**Why it's superior**:
`colibrì` currently uses parallel I/O workers and `compat_pread` (which under Windows uses `OVERLAPPED` but blocks synchronously on `ReadFile`). A true async approach using `io_uring` or Windows IOCP allows the OS to independently stream data from the NVMe drive directly into a pinned RAM buffer while the CPU/GPU are actively computing the current layer. Double-buffering prevents the compute pipeline from stalling on I/O.

**Implementation (Pseudo-code)**:
```c
// 1. Structure for double-buffering
typedef struct {
    int fd;
    void* buf[2];          // 2x Max layer size, aligned to 4096 (O_DIRECT)
    int active_idx;
    int prefetch_idx;
#ifdef _WIN32
    HANDLE iocp;           // Windows IO Completion Port
    OVERLAPPED ov;
#else
    struct io_uring ring;  // Linux io_uring
#endif
} ffn_async_buffer;

// 2. Async Dispatch
void ffn_async_prefetch(ffn_async_buffer* ab, uint64_t offset, size_t size) {
#ifdef _WIN32
    ab->ov.Offset = offset & 0xFFFFFFFF;
    ab->ov.OffsetHigh = offset >> 32;
    ReadFile((HANDLE)_get_osfhandle(ab->fd), ab->buf[ab->prefetch_idx], size, NULL, &ab->ov);
#else
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ab->ring);
    io_uring_prep_read(sqe, ab->fd, ab->buf[ab->prefetch_idx], size, offset);
    io_uring_submit(&ab->ring);
#endif
}

// 3. Async Wait & Swap
void ffn_async_swap(ffn_async_buffer* ab) {
#ifdef _WIN32
    DWORD bytes;
    GetOverlappedResult((HANDLE)_get_osfhandle(ab->fd), &ab->ov, &bytes, TRUE);
#else
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ab->ring, &cqe);
    io_uring_cqe_seen(&ab->ring, cqe);
#endif
    // Ping-pong buffers
    int tmp = ab->active_idx;
    ab->active_idx = ab->prefetch_idx;
    ab->prefetch_idx = tmp;
}
```

### Step 3: Decoupled Attention & FFN Pipeline Overlap

**Why it's superior**:
By restructuring the layer loop, we can execute the GPU Attention block, CPU FFN block, and SSD I/O concurrently. This pipeline hides the SSD read latency of FFN weights behind the dense matrix multiplications happening on the GPU.

**Implementation (Pseudo-code)**:
```c
// Inside the main token generation loop for(layer = 0; layer < N; layer++)

// 1. Kick off async prefetch for the NEXT layer's FFN weights
if (layer + 1 < N) {
    ffn_async_prefetch(async_buf, ffn_offsets[layer + 1], ffn_sizes[layer + 1]);
}

// 2. GPU computes Attention for CURRENT layer
// (This runs asynchronously on the CUDA stream)
coli_cuda_attention_absorb_batch_dev(..., d_residual);

// 3. Wait for PREVIOUS layer's FFN prefetch to complete (if layer > 0)
// For layer 0, we synchronously loaded it before the loop.
ffn_async_swap(async_buf);

// 4. Synchronize GPU (Wait for Attention to finish)
cudaStreamSynchronize(stream);

// 5. CPU directly reads/writes residual via ReBAR mapped memory
// Compute FFN using CPU RAM weights and ReBAR residual stream
llm_compute_ffn_cpu(async_buf->buf[async_buf->active_idx], h_residual);

// Output of FFN is written directly to h_residual, which is mirrored in VRAM.
// Next layer loop begins immediately without DMA copy overhead.
```

## Conclusion
By adopting the PoC's **ReBAR memory mapping**, **io_uring/IOCP double-buffering**, and **Decoupled execution pipeline**, `colibrì` will maximize utilization of all system buses (PCIe to GPU, PCIe to NVMe) simultaneously. This is a superior upgrade because it changes the execution from a sequential block-and-wait pattern to a highly concurrent streaming pipeline, which is the only way to run massive >100B parameter models at acceptable generation speeds on VRAM-constrained hardware.
