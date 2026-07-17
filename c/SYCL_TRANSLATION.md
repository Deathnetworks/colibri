# CUDA to SYCL Translation Document

This document outlines the required elements to translate the `backend_cuda.cu` runtime into a fully functional `backend_sycl.cpp` runtime for Intel/AMD GPUs via SYCL.

## 1. Context & Memory Management
*   **`coli_sycl_init` / `coli_sycl_shutdown`**: Detect available SYCL devices (GPUs), allocate `sycl::queue` instances, and track contexts.
*   **`coli_sycl_mem_info`**: Query `sycl::info::device::global_mem_size` and `ext_intel_free_memory` to report real VRAM availability, ensuring the planner correctly utilizes the GPU.
*   **`coli_sycl_tensor_upload` / `update` / `free`**: Allocate device memory via `sycl::malloc_device`, upload weights/scales, and implement the INT4 packed weight sign conversion (`offset_to_signed_s4` equivalent).
*   **Pipeline Memory (`pipe_alloc`, `pipe_scratch`, `pipe_copy`, etc.)**: Map `cudaMalloc`, `cudaMemcpy`, and device-to-device transfers to `sycl::malloc_device` and `queue::memcpy`.

## 2. Element-wise Compute Primitives
*   **`coli_sycl_pipe_rmsnorm`**: Calculate variance, inverse square root, and scale rows. Needs a SYCL reduction or simple sequential/blocked sum.
*   **`coli_sycl_pipe_rope`**: Apply Rotary Positional Embeddings. Maps straight to a 1D `sycl::parallel_for`.
*   **`coli_sycl_pipe_silu_mul`**: Standard SiLU activation and multiplication. Maps to `sycl::parallel_for` using `sycl::exp`.
*   **`coli_sycl_pipe_add` / `rows_add`**: Vector addition kernels.

## 3. Matrix Multiplication (GEMM)
*   **`coli_sycl_matmul`**: The core operation. Needs to support FP32, INT8, INT4, and INT2 weights against FP32 activations.
    *   *Work to be done*: Implement a SYCL kernel that spawns a thread per output feature, looping over the input dimension, reading packed weights (unpacking INT4/INT2), multiplying by the activation, and accumulating. Initially, a naive `parallel_for` will be used instead of optimized Shared Memory/Tensor Cores.

## 4. Fused MLPs
*   **`coli_sycl_expert_mlp`**: Fuses `gate`, `up` projections, `silu_mul`, and `down` projection.
    *   *Work to be done*: Sequentially queue the three matrix multiplications and the element-wise `silu_mul` on the SYCL queue.
*   **`coli_sycl_shared_mlp_w4a16`**: Specialized for INT4.
    *   *Work to be done*: Fallback to standard GEMM sequence or write a unified SYCL kernel.
*   **`coli_sycl_expert_group`**: Batched execution of multiple experts for Mixture-of-Experts.
    *   *Work to be done*: Loop through the provided experts and launch a GEMM for each segment of the activation tensor based on the router's row assignments.

## 5. Attention Primitives
*   **`coli_sycl_attention_absorb`**: Fused RoPE + QK/V projection for Multi-Head Latent Attention (MLA).
*   **`coli_sycl_attention_absorb_batch` / `project_batch`**: Batched versions that keep KV caching entirely on the device.
    *   *Work to be done*: Port the `attention_absorb_batch_kernel` logic, which applies RoPE, calculates attention scores (softmax), and projects against the latent state. A basic SYCL kernel translating the thread-block logic (using groups and local memory barriers) will be implemented to ensure functional correctness.
