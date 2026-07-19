# XPU Support and Intel Arc Pro B70 Optimization Plan

## 1. Introduction
This document outlines the strategy for adding XPU (Intel oneAPI/SYCL) support to the `colibrì` engine. By providing an XPU backend, the engine can efficiently execute on Intel discrete GPUs, such as the Intel Arc Pro B70 (based on the Xe2 architecture with 32 Xe-cores).

## 2. Translating CUDA/Metal Kernels to XPU (SYCL)

### 2.1 Backend Architecture
A new `backend_sycl.cpp` / `backend_sycl.h` will be implemented, mirroring the API of `backend_cuda.h`.
SYCL provides a single-source programming model in C++ that can target both CPU and GPU.

### 2.2 Kernel Translation Strategy
- **Grid/Block to ND-Range:** CUDA's `dim3 grid, block` and `blockIdx`, `threadIdx` map directly to SYCL's `nd_range` and `nd_item`.
- **Shared Memory:** CUDA `__shared__` memory maps to SYCL `local_accessor`.
- **Warp-level Primitives:** CUDA's `__shfl_down_sync` maps to SYCL's `sub_group` operations (e.g., `sycl::ext::oneapi::sub_group::shuffle_down`).
- **Atomic Operations:** CUDA `atomicAdd` maps to `sycl::atomic_ref`.

### 2.3 Example Translation: `silu_mul`
**CUDA:**
```cuda
__global__ void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = gate[i];
        gate[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}
```

**SYCL (XPU):**
```cpp
void silu_mul(sycl::queue& q, float* gate, const float* up, size_t n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        float x = gate[i];
        gate[i] = (x / (1.0f + sycl::exp(-x))) * up[i];
    });
}
```

## 3. Specific Optimizations for Intel Arc Pro B70 (32 Xe2 Cores)

The Arc Pro B70 features the newer Xe2 architecture. To maximize performance on this hardware:

### 3.1 Leverage Xe Matrix Extensions (XMX)
Xe2 cores possess advanced XMX engines capable of high-throughput matrix multiplication.
- **DPAS (Dot Product Accumulate Systolic):** Utilize SYCL's joint matrix extensions (`sycl::ext::oneapi::experimental::matrix`) to map the dense `quant_matmul` and `attention_absorb_batch_kernel` directly to XMX hardware.
- **INT4/INT8 Support:** The `colibrì` engine makes heavy use of INT4 (Q4) and INT8 weights. XMX on Xe2 is highly optimized for INT8 and INT4 DPAS instructions, offering immense TOPS. We must implement the dequantization step so that it directly feeds the XMX systolic arrays.

### 3.2 Sub-Group Sizes (SIMD Width)
Xe2 GPUs natively execute with sub-group sizes (SIMD width) of 16 or 32.
- Ensure that workgroup sizes are multiples of the sub-group size (typically 16 or 32 on Xe2).
- Explicitly request the optimal sub-group size in kernels via `reqd_sub_group_size(32)` to avoid performance cliffs caused by the compiler selecting a suboptimal SIMD width.

### 3.3 Large L2 Cache and SLM (Shared Local Memory)
The B70 has significant L2 cache and fast SLM.
- Maximize the use of SLM for KV-cache tiling during attention computation.
- Xe2 improves SLM bandwidth and atomic performance. Group reductions (e.g., during RMSNorm and Softmax) should heavily utilize SLM-based tree reductions combined with sub-group shuffles.

### 3.4 Command Queue and Submission Optimizations
- Use `sycl::property::queue::in_order()` where dependency graphs are strictly linear to reduce SYCL runtime overhead.
- For the Decoupled Attention/FFN pipeline, use independent out-of-order queues to ensure the SSD to VRAM (via ReBAR) memory copies overlap perfectly with XMX compute kernels.

## 4. ReBAR Integration on Intel Arc
Intel GPUs heavily rely on Resizable BAR (ReBAR) for performance. As specified in the Decoupling Plan, SYCL Unified Shared Memory (USM) must be used.
- Allocate residual buffers with `sycl::malloc_host` (pinned) or `sycl::malloc_device` and allow direct CPU writes.
- SYCL provides direct mechanisms to check for ReBAR capacity and optimize USM allocations accordingly.
