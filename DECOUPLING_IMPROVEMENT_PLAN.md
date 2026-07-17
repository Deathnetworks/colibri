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

**Implementation**:
1.  **In `c/backend_cuda.cu`**: Introduce a new allocation function that exposes mapped memory allocation. We need to allocate memory using `cudaHostAlloc` with the `cudaHostAllocMapped` flag to allocate zero-copy pinned memory (or ReBAR mapped device memory).
    ```c
    extern "C" void* coli_cuda_alloc_mapped(size_t bytes, void** device_ptr) {
        void* host_ptr;
        // Allocate zero-copy pinned memory (or ReBAR mapped device memory)
        cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped);
        cudaHostGetDevicePointer(device_ptr, host_ptr, 0);
        return host_ptr;
    }
    ```
2.  **In `c/glm.c`**: Update the residual stream buffers to use this new mapped memory allocation instead of standard heap allocation + `cudaMemcpy`.
    ```c
    void* d_residual;
    float* h_residual = coli_cuda_alloc_mapped(S * D * sizeof(float), &d_residual);
    ```
3.  **Data Transfer**: Replace `cudaMemcpyAsync` calls for these specific residual buffers with direct `memcpy`, allowing the CPU to write directly to VRAM via ReBAR.
    ```c
    // CPU writes directly to VRAM via ReBAR, visible to GPU immediately.
    memcpy(h_residual, cpu_ffn_output, size);
    ```

### Step 2: Double-Buffered True Async I/O (DirectStorage & io_uring)

**Why it's superior**:
`colibrì` currently uses parallel I/O workers and `compat_pread` (which under Windows uses `OVERLAPPED` but blocks synchronously on `ReadFile`). A true async approach using `io_uring` or Windows IOCP allows the OS to independently stream data from the NVMe drive directly into a pinned RAM buffer while the CPU/GPU are actively computing the current layer. Double-buffering prevents the compute pipeline from stalling on I/O.

**Implementation**:
1.  **Async Double-Buffer Engine (`ffn_async_buffer`)**: Create a structure to manage the double-buffering state, holding file descriptors, aligned buffers (for `O_DIRECT`), active/prefetch indices, and OS-specific async I/O primitives (`io_uring` for Linux, `OVERLAPPED`/`HANDLE` for Windows).
2.  **Initialization and Allocation**: Implement `ffn_async_init` to set up the `O_DIRECT` file descriptor and allocate two aligned buffers (e.g., using `aligned_alloc_4k` to ensure 4096-byte alignment required by `O_DIRECT`).
3.  **Async Dispatch (`ffn_async_prefetch`)**: Implement non-blocking read dispatch. On Linux, prepare an `io_uring_sqe` with `io_uring_prep_read` (or `io_uring_prep_read_fixed`) and submit it. On Windows, use `ReadFile` with the `OVERLAPPED` structure, ensuring it returns immediately (e.g., returning `ERROR_IO_PENDING`).
4.  **Async Wait & Swap (`ffn_async_swap`)**: Implement the synchronization point to wait for the prefetched buffer to finish loading. Use `io_uring_wait_cqe` on Linux or `GetOverlappedResult` on Windows. Once the read completes, swap the `active_idx` and `prefetch_idx` to ping-pong the buffers.

### Step 3: Decoupled Attention & FFN Pipeline Overlap

**Why it's superior**:
By restructuring the layer loop, we can execute the GPU Attention block, CPU FFN block, and SSD I/O concurrently. This pipeline hides the SSD read latency of FFN weights behind the dense matrix multiplications happening on the GPU.

**Implementation**:
1.  **Restructure the Main Generation Loop**: In the token generation loop in `c/glm.c` (specifically around the block processing loop, likely inside a function like `forward` or similar), reorganize the operations to allow overlapping.
2.  **Concurrency Flow**:
    *   **Start of Layer N**: Immediately dispatch the async prefetch for Layer N+1's FFN weights (`ffn_async_prefetch`).
    *   **GPU Attention**: Launch the GPU Attention computation for Layer N asynchronously on the CUDA stream.
    *   **Wait for Layer N FFN**: Wait for the prefetch of Layer N's FFN weights (which was initiated in the *previous* layer's iteration) to complete (`ffn_async_swap`).
    *   **Synchronize GPU**: Wait for the GPU Attention computation for Layer N to finish (`cudaStreamSynchronize`).
    *   **CPU FFN**: Execute the FFN block on the CPU (`llm_compute_ffn_cpu`), reading the weights from the active async buffer and writing the output directly to the ReBAR-mapped residual stream buffer.
3.  **CPU FFN Implementation**: Implement the CPU-based FFN computation (`llm_compute_ffn_cpu`). This will involve porting the logic from `llama.cpp-PoC` (`src/llama-ffn-local.cpp`), handling dequantization on-the-fly (e.g., Q4 to F32) if necessary, applying RMS normalization, the gate/up projections, the SiLU activation, and the down projection.

## 3. Potential Challenges & Considerations

*   **Dequantization**: If weights are quantized (e.g., INT4/INT8), the CPU FFN compute will need to dequantize them on-the-fly, which adds CPU overhead. Ensuring this is highly optimized is critical.
*   **CPU Bottleneck**: While I/O is hidden and GPU is utilized for attention, the FFN compute is shifted entirely to the CPU. For very large models, CPU FLOPs might become the new bottleneck if not parallelized effectively (e.g., using OpenMP).
*   **Cross-Platform ReBAR**: Ensuring robust fallback mechanisms if ReBAR/zero-copy mapping fails or is unavailable on a specific system.
*   **DirectStorage/O_DIRECT alignment**: Strict memory alignment (usually 4KB) is required for these APIs. Care must be taken to ensure all buffers passed to async I/O calls meet these requirements.

## Conclusion
By adopting the PoC's **ReBAR memory mapping**, **io_uring/IOCP double-buffering**, and **Decoupled execution pipeline**, `colibrì` will maximize utilization of all system buses (PCIe to GPU, PCIe to NVMe) simultaneously. This is a superior upgrade because it changes the execution from a sequential block-and-wait pattern to a highly concurrent streaming pipeline, which is the only way to run massive >100B parameter models at acceptable generation speeds on VRAM-constrained hardware.
