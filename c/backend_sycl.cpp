#include "backend_sycl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#if defined(__SYCL_COMPILER_VERSION) || defined(__INTEL_LLVM_COMPILER)
#include <sycl/sycl.hpp>
#else
// Dummy SYCL definitions for platforms where SYCL compiler is absent but we want to build the shim
namespace sycl {
    struct queue {
        void wait() {}
        template<typename Range, typename Func>
        void parallel_for(Range r, Func f) {}
    };
    template<int dim> struct range { range(size_t) {} };
    template<int dim> struct id { size_t operator[](int) const { return 0; } };
    float exp(float x) { return std::exp(x); }
    template<typename T> void* malloc_device(size_t sz, queue& q) { return malloc(sz); }
    template<typename T> void* malloc_host(size_t sz, queue& q) { return malloc(sz); }
    void free(void* ptr, queue& q) { ::free(ptr); }
}
#endif

// Stub implementation for now. The plan specifically highlights silu_mul.

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

static sycl::queue* g_q = nullptr;

int coli_sycl_init(const int *devices, int count) {
    if (!g_q) g_q = new sycl::queue();
    return 1;
}

void coli_sycl_shutdown(void) {
    if (g_q) {
        delete g_q;
        g_q = nullptr;
    }
}

int coli_sycl_device_count(void) { return 1; }
int coli_sycl_device_at(int index) { return 0; }
int coli_sycl_mem_info(int device, size_t *free_bytes, size_t *total_bytes) { *free_bytes = 1024*1024*1024; *total_bytes = 1024*1024*1024; return 1; }
void coli_sycl_stats(int device, size_t *tensor_count, size_t *tensor_bytes) { *tensor_count = 0; *tensor_bytes = 0; }
void coli_sycl_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows, double *h2d_ms, double *kernel_ms, double *d2h_ms) { *calls=0; *experts=0; *rows=0; *h2d_ms=0; *kernel_ms=0; *d2h_ms=0; }
int coli_sycl_tensor_upload(ColiSyclTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device) { return 0; }
int coli_sycl_matmul(ColiSyclTensor **tensor, float *y, const float *x, const void *weights, const float *scales, int fmt, int S, int I, int O, int device) { return 0; }
int coli_sycl_expert_mlp(ColiSyclTensor *gate, ColiSyclTensor *up, ColiSyclTensor *down, float *y, const float *x, int S) { return 0; }
int coli_sycl_shared_mlp_w4a16(ColiSyclTensor *gate, ColiSyclTensor *up, ColiSyclTensor *down, float *y, const float *x, int S) { return 0; }
int coli_sycl_expert_group(ColiSyclTensor *const *gates, ColiSyclTensor *const *ups, ColiSyclTensor *const *downs, const int *rows, int count, float *y, const float *x) { return 0; }
int coli_sycl_attention_absorb(ColiSyclTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int H,int Q, int R,int V,int K,int T,float attention_scale) { return 0; }
int coli_sycl_attention_absorb_batch(ColiSyclTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale) { return 0; }
int coli_sycl_attention_project_batch(ColiSyclTensor *kv_b,ColiSyclTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale) { return 0; }
void coli_sycl_tensor_free(ColiSyclTensor *tensor) {}
size_t coli_sycl_tensor_bytes(const ColiSyclTensor *tensor) { return 0; }
int coli_sycl_tensor_device(const ColiSyclTensor *tensor) { return -1; }
int coli_sycl_tensor_update(ColiSyclTensor *tensor, const void *weights, const float *scales) { return 0; }
float *coli_sycl_pipe_scratch(int device,int slot,size_t bytes) { return NULL; }
void *coli_sycl_pipe_alloc(int device,size_t bytes) { return NULL; }
void coli_sycl_pipe_free(int device,void *p) {}
int coli_sycl_pipe_upload(int device,void *dst,const void *src,size_t bytes) { return 0; }
int coli_sycl_pipe_download(int device,const void *src,void *dst,size_t bytes) { return 0; }
int coli_sycl_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps) { return 0; }
int coli_sycl_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta) { return 0; }

int coli_sycl_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n) {
    if (!g_q) return 0;

    // Example from XPU_SUPPORT_PLAN.md translated properly using actual SYCL queue and lambda
    g_q->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        float x = gate_dev[i[0]];
        gate_dev[i[0]] = (x / (1.0f + sycl::exp(-x))) * up_dev[i[0]];
    });
    g_q->wait();
    return 1;
}

int coli_sycl_pipe_add(int device,float *x_dev,const float *t_dev,size_t n) { return 0; }
int coli_sycl_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D) { return 0; }
int coli_sycl_pipe_gemm(ColiSyclTensor *t,float *y_dev,const float *x_dev,int S) { return 0; }
int coli_sycl_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride) { return 0; }
int coli_sycl_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta) { return 0; }
int coli_sycl_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height) { return 0; }
int coli_sycl_attention_project_batch_dev(ColiSyclTensor *kv_b,ColiSyclTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) { return 0; }
int coli_sycl_attention_absorb_batch_dev(ColiSyclTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) { return 0; }
int coli_sycl_attention_absorb_kvdev(ColiSyclTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale) { return 0; }
int coli_sycl_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes) { return 0; }
int coli_sycl_attention_project_batch_dev_out(ColiSyclTensor *kv_b,ColiSyclTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) { return 0; }
int coli_sycl_pipe_sync(int device) {
    if (g_q) { g_q->wait(); return 1; }
    return 0;
}
