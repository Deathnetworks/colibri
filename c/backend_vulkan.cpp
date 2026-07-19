#include "backend_vulkan.h"
#include <cstdio>
#include <cstdlib>

// Stub implementation for Vulkan backend.
// A full Vulkan compute backend requires setting up instances, devices, compute queues,
// command buffers, and submitting spir-v shaders.
// For now, this serves as the linkage point and stub similar to the SYCL initial setup.

int coli_vulkan_init(const int *devices, int count) { return 1; }
void coli_vulkan_shutdown(void) {}
int coli_vulkan_device_count(void) { return 1; }
int coli_vulkan_device_at(int index) { return 0; }
int coli_vulkan_mem_info(int device, size_t *free_bytes, size_t *total_bytes) { *free_bytes=1024*1024*1024; *total_bytes=1024*1024*1024; return 1; }
void coli_vulkan_stats(int device, size_t *tensor_count, size_t *tensor_bytes) { *tensor_count=0; *tensor_bytes=0; }
void coli_vulkan_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows, double *h2d_ms, double *kernel_ms, double *d2h_ms) { *calls=0; *experts=0; *rows=0; *h2d_ms=0; *kernel_ms=0; *d2h_ms=0; }
int coli_vulkan_tensor_upload(ColiVulkanTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device) { return 0; }
int coli_vulkan_matmul(ColiVulkanTensor **tensor, float *y, const float *x, const void *weights, const float *scales, int fmt, int S, int I, int O, int device) { return 0; }
int coli_vulkan_expert_mlp(ColiVulkanTensor *gate, ColiVulkanTensor *up, ColiVulkanTensor *down, float *y, const float *x, int S) { return 0; }
int coli_vulkan_shared_mlp_w4a16(ColiVulkanTensor *gate, ColiVulkanTensor *up, ColiVulkanTensor *down, float *y, const float *x, int S) { return 0; }
int coli_vulkan_expert_group(ColiVulkanTensor *const *gates, ColiVulkanTensor *const *ups, ColiVulkanTensor *const *downs, const int *rows, int count, float *y, const float *x) { return 0; }
int coli_vulkan_attention_absorb(ColiVulkanTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int H,int Q, int R,int V,int K,int T,float attention_scale) { return 0; }
int coli_vulkan_attention_absorb_batch(ColiVulkanTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale) { return 0; }
int coli_vulkan_attention_project_batch(ColiVulkanTensor *kv_b,ColiVulkanTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale) { return 0; }
int coli_vulkan_attention_project_ragged(ColiVulkanTensor *kv_b,ColiVulkanTensor *o_proj, float *out,const float *q,const void *const *keys, const float *const *latent,const float *const *rope, const int *lengths,int S,int H,int Q,int R,int V,int K,int T,float scale) { return 0; }
void coli_vulkan_tensor_free(ColiVulkanTensor *tensor) {}
size_t coli_vulkan_tensor_bytes(const ColiVulkanTensor *tensor) { return 0; }
int coli_vulkan_tensor_device(const ColiVulkanTensor *tensor) { return -1; }
int coli_vulkan_tensor_update(ColiVulkanTensor *tensor, const void *weights, const float *scales) { return 0; }
float *coli_vulkan_pipe_scratch(int device,int slot,size_t bytes) { return NULL; }
void *coli_vulkan_pipe_alloc(int device,size_t bytes) { return NULL; }
void coli_vulkan_pipe_free(int device,void *p) {}
int coli_vulkan_pipe_upload(int device,void *dst,const void *src,size_t bytes) { return 0; }
int coli_vulkan_pipe_download(int device,const void *src,void *dst,size_t bytes) { return 0; }
int coli_vulkan_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps) { return 0; }
int coli_vulkan_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta) { return 0; }
int coli_vulkan_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n) { return 0; }
int coli_vulkan_pipe_add(int device,float *x_dev,const float *t_dev,size_t n) { return 0; }
int coli_vulkan_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D) { return 0; }
int coli_vulkan_pipe_gemm(ColiVulkanTensor *t,float *y_dev,const float *x_dev,int S) { return 0; }
int coli_vulkan_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride) { return 0; }
int coli_vulkan_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta) { return 0; }
int coli_vulkan_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height) { return 0; }
int coli_vulkan_attention_project_batch_dev(ColiVulkanTensor *kv_b,ColiVulkanTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) { return 0; }
int coli_vulkan_attention_absorb_batch_dev(ColiVulkanTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) { return 0; }
int coli_vulkan_attention_absorb_kvdev(ColiVulkanTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale) { return 0; }
int coli_vulkan_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes) { return 0; }
int coli_vulkan_attention_project_batch_dev_out(ColiVulkanTensor *kv_b,ColiVulkanTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) { return 0; }
int coli_vulkan_pipe_sync(int device) { return 1; }

extern "C" COLI_VULKAN_DLLEXPORT void* coli_vulkan_alloc_mapped(size_t bytes, void** device_ptr) {
    return nullptr; // Stub for Vulkan
}

extern "C" COLI_VULKAN_DLLEXPORT void coli_vulkan_free_mapped(void* host_ptr) {
}
