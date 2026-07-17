#ifndef COLIBRI_BACKEND_SYCL_H
#define COLIBRI_BACKEND_SYCL_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(COLI_SYCL_BUILDING_DLL)
#define COLI_SYCL_DLLEXPORT __declspec(dllexport)
#else
#define COLI_SYCL_DLLEXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_SYCL_MAX_DEVICES 16

/* Re-use ColiCudaTensor typedef as an opaque pointer for SYCL to maintain ABI compatibility with backend_loader.c */
typedef struct ColiCudaTensor ColiSyclTensor;
typedef struct ColiCudaTensor ColiCudaTensor;
/* To compile this standalone, we need to forward declare ColiCudaTensor */
typedef struct ColiCudaTensor ColiCudaTensor;


COLI_SYCL_DLLEXPORT int coli_sycl_init(const int *devices, int count);
COLI_SYCL_DLLEXPORT void coli_sycl_shutdown(void);
COLI_SYCL_DLLEXPORT int coli_sycl_device_count(void);
COLI_SYCL_DLLEXPORT int coli_sycl_device_at(int index);
COLI_SYCL_DLLEXPORT int coli_sycl_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
COLI_SYCL_DLLEXPORT void coli_sycl_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_SYCL_DLLEXPORT void coli_sycl_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);

COLI_SYCL_DLLEXPORT int coli_sycl_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

COLI_SYCL_DLLEXPORT int coli_sycl_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device);

COLI_SYCL_DLLEXPORT int coli_sycl_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

COLI_SYCL_DLLEXPORT int coli_sycl_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up,
                               ColiCudaTensor *down, float *y,
                               const float *x, int S);

COLI_SYCL_DLLEXPORT int coli_sycl_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);

COLI_SYCL_DLLEXPORT int coli_sycl_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

COLI_SYCL_DLLEXPORT int coli_sycl_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q,
                                     const float *latent,const float *rope,int S,
                                     int H,int Q,int R,int V,int K,int T,
                                     float attention_scale);

COLI_SYCL_DLLEXPORT int coli_sycl_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
                                      float *out,const float *q,const float *latent,
                                      const float *rope,int S,int H,int Q,int R,
                                      int V,int K,int T,float attention_scale);

COLI_SYCL_DLLEXPORT void coli_sycl_tensor_free(ColiCudaTensor *tensor);
COLI_SYCL_DLLEXPORT size_t coli_sycl_tensor_bytes(const ColiCudaTensor *tensor);
COLI_SYCL_DLLEXPORT int coli_sycl_tensor_device(const ColiCudaTensor *tensor);
COLI_SYCL_DLLEXPORT int coli_sycl_tensor_update(ColiCudaTensor *tensor,
                            const void *weights, const float *scales);

COLI_SYCL_DLLEXPORT float *coli_sycl_pipe_scratch(int device,int slot,size_t bytes);
COLI_SYCL_DLLEXPORT void *coli_sycl_pipe_alloc(int device,size_t bytes);
COLI_SYCL_DLLEXPORT void coli_sycl_pipe_free(int device,void *p);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_upload(int device,void *dst,const void *src,size_t bytes);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_download(int device,const void *src,void *dst,size_t bytes);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                           const float *w_dev,int S,int D,float eps);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows,
                        int stride,int offset,int R,int heads,float theta);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_add(int device,float *x_dev,const float *t_dev,size_t n);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                            const int *rows_dev,int nrows,int D);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                             const float *w_dev,int S,int D,float eps,
                             int xstride,int ystride);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                             int stride,int offset,int R,int heads,float theta);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                          int spitch,int width,int height);
COLI_SYCL_DLLEXPORT int coli_sycl_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_SYCL_DLLEXPORT int coli_sycl_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_SYCL_DLLEXPORT int coli_sycl_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                             const float *src,size_t bytes);
COLI_SYCL_DLLEXPORT int coli_sycl_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_SYCL_DLLEXPORT int coli_sycl_pipe_sync(int device);

#ifdef __cplusplus
}
#endif

#endif
