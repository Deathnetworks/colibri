/* backend_loader.c — Windows runtime loader for coli_cuda.dll.
 *
 * Why this exists: the engine is built with MinGW-w64 (gcc), but CUDA kernels
 * must be compiled with MSVC + nvcc. We cannot link a CUDA .o into a gcc binary
 * reliably across the MSVC/MinGW ABI, and nvcc requires cl.exe as its host
 * compiler. The clean cross-toolchain split is: build the CUDA backend into a
 * standalone coli_cuda.dll with nvcc+MSVC, then load it here at runtime via
 * LoadLibrary/GetProcAddress. The host (glm.exe) never links cudart directly.
 *
 * On Linux this file is not compiled (the Makefile links backend_cuda.o
 * directly). On Windows, when COLI_CUDA is defined, glm.c calls the
 * coli_cuda_* wrappers below, which forward through function pointers resolved
 * from the DLL at first use. If the DLL is absent, every call safely returns
 * the "not initialized" sentinel (0 / no-op) and the engine falls back to CPU.
 *
 * ABI note: ColiCudaTensor* is opaque to the host (it stores the pointer,
 * never dereferences it), so the MSVC-allocated struct is safe to pass across
 * the boundary as an opaque handle. All scalar types (int, size_t, pointers)
 * agree between MSVC and MinGW-w64 on x86-64.
 */
#ifdef _WIN32

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <windows.h>

#include "backend_cuda.h"
#include "backend_sycl.h"
#include "backend_vulkan.h"

/* Function-pointer typedefs matching each exported symbol. */
typedef int            (*fn_init)(const int *devices, int count);
typedef void           (*fn_shutdown)(void);
typedef int            (*fn_device_count)(void);
typedef int            (*fn_device_at)(int index);
typedef int            (*fn_mem_info)(int device, size_t *free_bytes, size_t *total_bytes);
typedef void           (*fn_stats)(int device, size_t *tensor_count, size_t *tensor_bytes);
typedef void           (*fn_group_stats)(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                         double *h2d_ms, double *kernel_ms, double *d2h_ms);
typedef int            (*fn_expert_mlp)(ColiCudaTensor *gate, ColiCudaTensor *up,
                                        ColiCudaTensor *down, float *y, const float *x, int S);
typedef int            (*fn_expert_group)(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                                          ColiCudaTensor *const *downs, const int *rows, int count,
                                          float *y, const float *x);
typedef int            (*fn_attention_absorb)(ColiCudaTensor *kv_b, float *ctx, const float *q,
                                              const float *latent, const float *rope, int H, int Q,
                                              int R, int V, int K, int T, float attention_scale);
typedef int            (*fn_tensor_upload)(ColiCudaTensor **tensor, const void *weights,
                                           const float *scales, int fmt, int I, int O, int device);
typedef int            (*fn_matmul)(ColiCudaTensor **tensor, float *y, const float *x,
                                    const void *weights, const float *scales,
                                    int fmt, int S, int I, int O, int device);
typedef void           (*fn_tensor_free)(ColiCudaTensor *tensor);
typedef size_t         (*fn_tensor_bytes)(const ColiCudaTensor *tensor);
typedef int            (*fn_tensor_device)(const ColiCudaTensor *tensor);

/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */


/* --- #111 GPU resident pipeline additions (matched to backend_cuda.h) --- */
typedef int (*fn_attention_absorb_batch)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale);
typedef int (*fn_attention_absorb_batch_dev)(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_absorb_kvdev)(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale);
typedef int (*fn_attention_project_batch)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale);
typedef int (*fn_attention_project_batch_dev)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_attention_project_batch_dev_out)(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale);
typedef int (*fn_pipe_add)(int device,float *x_dev,const float *t_dev,size_t n);
typedef void * (*fn_pipe_alloc)(int device,size_t bytes);
typedef int (*fn_pipe_copy2d)(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height);
typedef int (*fn_pipe_download)(int device,const void *src,void *dst,size_t bytes);
typedef void (*fn_pipe_free)(int device,void *p);
typedef int (*fn_pipe_gemm)(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
typedef int (*fn_pipe_peer_copy)(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes);
typedef int (*fn_pipe_rmsnorm)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps);
typedef int (*fn_pipe_rmsnorm_s)(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride);
typedef int (*fn_pipe_rope)(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rope_base)(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta);
typedef int (*fn_pipe_rows_add)(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D);
typedef float * (*fn_pipe_scratch)(int device,int slot,size_t bytes);
typedef int (*fn_pipe_silu_mul)(int device,float *gate_dev,const float *up_dev,size_t n);
typedef int (*fn_pipe_sync)(int device);
typedef int (*fn_pipe_upload)(int device,void *dst,const void *src,size_t bytes);
typedef int (*fn_shared_mlp_w4a16)(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S);
typedef int (*fn_tensor_update)(ColiCudaTensor *tensor, const void *weights, const float *scales);

/* Resolved pointers, plus a flag so we attempt the load at most once. */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_init            init;
    fn_shutdown        shutdown;
    fn_device_count    device_count;
    fn_device_at       device_at;
    fn_mem_info        mem_info;
    fn_stats           stats;
    fn_group_stats     group_stats;
    fn_expert_mlp      expert_mlp;
    fn_expert_group    expert_group;
    fn_attention_absorb attention_absorb;
    fn_tensor_upload   tensor_upload;
    fn_matmul          matmul;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;

    fn_attention_absorb_batch attention_absorb_batch;
    fn_attention_absorb_batch_dev attention_absorb_batch_dev;
    fn_attention_absorb_kvdev attention_absorb_kvdev;
    fn_attention_project_batch attention_project_batch;
    fn_attention_project_batch_dev attention_project_batch_dev;
    fn_attention_project_batch_dev_out attention_project_batch_dev_out;
    fn_pipe_add pipe_add;
    fn_pipe_alloc pipe_alloc;
    fn_pipe_copy2d pipe_copy2d;
    fn_pipe_download pipe_download;
    fn_pipe_free pipe_free;
    fn_pipe_gemm pipe_gemm;
    fn_pipe_peer_copy pipe_peer_copy;
    fn_pipe_rmsnorm pipe_rmsnorm;
    fn_pipe_rmsnorm_s pipe_rmsnorm_s;
    fn_pipe_rope pipe_rope;
    fn_pipe_rope_base pipe_rope_base;
    fn_pipe_rows_add pipe_rows_add;
    fn_pipe_scratch pipe_scratch;
    fn_pipe_silu_mul pipe_silu_mul;
    fn_pipe_sync pipe_sync;
    fn_pipe_upload pipe_upload;
    fn_shared_mlp_w4a16 shared_mlp_w4a16;
    fn_tensor_update tensor_update;
    void* (*alloc_mapped)(size_t bytes, void** device_ptr);
    void (*free_mapped)(void* host_ptr);
} g_cuda;

/* Resolve the DLL and all 11 symbols. Returns 1 on success, 0 otherwise.
 * Idempotent: the first call (success or fail) sticks; later calls are no-ops
 * that return the cached result. The engine treats a 0 return as "CUDA
 * unavailable" and falls back to the CPU path without aborting. */


static int coli_cuda_load(void){
    if(g_cuda.loaded) return g_cuda.available;
    g_cuda.loaded = 1;

    /* Load coli_cuda.dll from the engine's OWN directory, by absolute path —
     * never a bare name. LoadLibraryA("coli_cuda.dll") searches the current
     * working directory (and, without SafeDllSearchMode, other writable dirs):
     * an attacker who drops a coli_cuda.dll where the user launches glm.exe (or
     * inside a downloaded model directory the user cd's into) would get their
     * DllMain run at load — classic DLL hijacking -> arbitrary code execution.
     * Resolving the path next to glm.exe and loading THAT specific file with
     * LOAD_WITH_ALTERED_SEARCH_PATH anchors both the DLL and its dependency
     * search to the trusted install directory instead of the CWD. */
    char dllpath[MAX_PATH];
    HMODULE hm = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&coli_cuda_load, &hm);
    DWORD mn = GetModuleFileNameA(hm, dllpath, (DWORD)sizeof(dllpath));
    if(mn > 0 && mn < sizeof(dllpath)){
        char *slash = strrchr(dllpath, '\\');
        if(slash && (size_t)(slash + 1 - dllpath) + sizeof("coli_cuda.dll") <= sizeof(dllpath)){
            strcpy(slash + 1, "coli_cuda.dll");
            g_cuda.dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    if(!g_cuda.dll){
        /* fallback (GetModuleFileNameA almost never fails): search only
         * in the application directory and System32, NEVER the CWD. */
        g_cuda.dll = LoadLibraryExA("coli_cuda.dll", NULL,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if(!g_cuda.dll){
        fprintf(stderr, "[CUDA] coli_cuda.dll not found; GPU tier disabled "
                        "(CPU path remains active).\n");
        return 0;
    }

    #define RESOLVE(name, type) \
        /* GetProcAddress returns FARPROC (void(*)(void)); casting it to a   \
         * specific function-pointer type is the standard LoadLibrary idiom. \
         * -Wcast-function-type flags it but it is safe: the DLL exported     \
         * the symbol with extern "C" and the exact signature we expect. */   \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_cuda.name = (type)GetProcAddress(g_cuda.dll, "coli_cuda_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_cuda.name){ \
            fprintf(stderr, "[CUDA] coli_cuda.dll missing symbol coli_cuda_" #name "\n"); \
            FreeLibrary(g_cuda.dll); g_cuda.dll=NULL; return 0; }

    RESOLVE(init,           fn_init)
    RESOLVE(shutdown,       fn_shutdown)
    RESOLVE(device_count,   fn_device_count)
    RESOLVE(device_at,      fn_device_at)
    RESOLVE(mem_info,       fn_mem_info)
    RESOLVE(stats,          fn_stats)
    RESOLVE(group_stats,    fn_group_stats)
    RESOLVE(expert_mlp,     fn_expert_mlp)
    RESOLVE(expert_group,   fn_expert_group)
    RESOLVE(attention_absorb, fn_attention_absorb)
    RESOLVE(tensor_upload,  fn_tensor_upload)
    RESOLVE(matmul,         fn_matmul)
    RESOLVE(tensor_free,    fn_tensor_free)
    RESOLVE(tensor_bytes,   fn_tensor_bytes)
    RESOLVE(tensor_device,  fn_tensor_device)

    RESOLVE(attention_absorb_batch, fn_attention_absorb_batch)
    RESOLVE(attention_absorb_batch_dev, fn_attention_absorb_batch_dev)
    RESOLVE(attention_absorb_kvdev, fn_attention_absorb_kvdev)
    RESOLVE(attention_project_batch, fn_attention_project_batch)
    RESOLVE(attention_project_batch_dev, fn_attention_project_batch_dev)
    RESOLVE(attention_project_batch_dev_out, fn_attention_project_batch_dev_out)
    RESOLVE(pipe_add, fn_pipe_add)
    RESOLVE(pipe_alloc, fn_pipe_alloc)
    RESOLVE(pipe_copy2d, fn_pipe_copy2d)
    RESOLVE(pipe_download, fn_pipe_download)
    RESOLVE(pipe_free, fn_pipe_free)
    RESOLVE(pipe_gemm, fn_pipe_gemm)
    RESOLVE(pipe_peer_copy, fn_pipe_peer_copy)
    RESOLVE(pipe_rmsnorm, fn_pipe_rmsnorm)
    RESOLVE(pipe_rmsnorm_s, fn_pipe_rmsnorm_s)
    RESOLVE(pipe_rope, fn_pipe_rope)
    RESOLVE(pipe_rope_base, fn_pipe_rope_base)
    RESOLVE(pipe_rows_add, fn_pipe_rows_add)
    RESOLVE(pipe_scratch, fn_pipe_scratch)
    RESOLVE(pipe_silu_mul, fn_pipe_silu_mul)
    RESOLVE(pipe_sync, fn_pipe_sync)
    RESOLVE(pipe_upload, fn_pipe_upload)
    RESOLVE(shared_mlp_w4a16, fn_shared_mlp_w4a16)
    RESOLVE(tensor_update, fn_tensor_update)
    #undef RESOLVE

    g_cuda.available = 1;
    return 1;
}

/* ---- Public wrappers: match backend_cuda.h signatures exactly.
 * Each forwards to the resolved pointer; if the DLL never loaded, return the
 * "not initialized" result the engine already handles (init returns 0, matmul
 * returns 0 so the caller marks the tensor cuda_failed and uses CPU). ---- */

int coli_cuda_init(const int *devices, int count){
    if(!coli_cuda_load()) return 0;
    return g_cuda.init(devices, count);
}

void coli_cuda_shutdown(void){
    if(g_cuda.available && g_cuda.shutdown) g_cuda.shutdown();
}

int coli_cuda_device_count(void){
    if(!g_cuda.available) return 0;
    return g_cuda.device_count();
}

int coli_cuda_device_at(int index){
    if(!g_cuda.available) return -1;
    return g_cuda.device_at(index);
}

int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_cuda.available){ if(free_bytes)*free_bytes=0; if(total_bytes)*total_bytes=0; return 0; }
    return g_cuda.mem_info(device, free_bytes, total_bytes);
}

void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    if(!g_cuda.available){ if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0; return; }
    g_cuda.stats(device, tensor_count, tensor_bytes);
}

void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(!g_cuda.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_cuda.group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_mlp(gate, up, down, y, x, S);
}

int coli_cuda_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs, const int *rows, int count,
                           float *y, const float *x){
    if(!g_cuda.available) return 0;
    return g_cuda.expert_group(gates, ups, downs, rows, count, y, x);
}

int coli_cuda_attention_absorb(ColiCudaTensor *kv_b, float *ctx, const float *q,
                               const float *latent, const float *rope, int H, int Q,
                               int R, int V, int K, int T, float attention_scale){
    if(!g_cuda.available) return 0;
    return g_cuda.attention_absorb(kv_b, ctx, q, latent, rope, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights,
                            const float *scales, int fmt, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_upload(tensor, weights, scales, fmt, I, O, device);
}

int coli_cuda_matmul(ColiCudaTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device){
    if(!g_cuda.available) return 0;
    return g_cuda.matmul(tensor, y, x, weights, scales, fmt, S, I, O, device);
}

void coli_cuda_tensor_free(ColiCudaTensor *tensor){
    if(g_cuda.available && g_cuda.tensor_free) g_cuda.tensor_free(tensor);
}

size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return 0;
    return g_cuda.tensor_bytes(tensor);
}

int coli_cuda_tensor_device(const ColiCudaTensor *tensor){
    if(!g_cuda.available) return -1;
    return g_cuda.tensor_device(tensor);
}

/* ---- #111 pipeline wrappers ---- */
void* coli_cuda_alloc_mapped(size_t bytes, void** device_ptr) {
    if(!g_cuda.dll) return NULL;
    return g_cuda.alloc_mapped(bytes, device_ptr);
}
void coli_cuda_free_mapped(void* host_ptr) {
    if(g_cuda.dll) g_cuda.free_mapped(host_ptr);
}



/* ---- #111 pipeline wrappers (see header for semantics) ---- */

int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch(kv_b, ctx, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_batch_dev(kv_b_shard, ctx_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_absorb_kvdev(kv_b, ctx, q, latent_dev, rope_dev, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch(kv_b, o_proj, out, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev(kv_b, o_proj, out, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_cuda.available){ return 0; }
    return g_cuda.attention_project_batch_dev_out(kv_b, o_proj, out_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_add(device, x_dev, t_dev, n);
}

void * coli_cuda_pipe_alloc(int device,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_alloc(device, bytes);
}

int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_copy2d(device, dst, dpitch, src, spitch, width, height);
}

int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_download(device, src, dst, bytes);
}

void coli_cuda_pipe_free(int device,void *p){
    if(!g_cuda.available){ return; }
    g_cuda.pipe_free(device, p);
}

int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_gemm(t, y_dev, x_dev, S);
}

int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_peer_copy(dst_dev, dst, src_dev, src, bytes);
}

int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm(device, y_dev, x_dev, w_dev, S, D, eps);
}

int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rmsnorm_s(device, y_dev, x_dev, w_dev, S, D, eps, xstride, ystride);
}

int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope(device, v_dev, pos_dev, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rope_base(device, v_dev, pos_base, rows, stride, offset, R, heads, theta);
}

int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_rows_add(device, x_dev, partial_dev, rows_dev, nrows, D);
}

float * coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    if(!g_cuda.available){ return NULL; }
    return g_cuda.pipe_scratch(device, slot, bytes);
}

int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_silu_mul(device, gate_dev, up_dev, n);
}

int coli_cuda_pipe_sync(int device){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_sync(device);
}

int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    if(!g_cuda.available){ return 0; }
    return g_cuda.pipe_upload(device, dst, src, bytes);
}

int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S){
    if(!g_cuda.available){ return 0; }
    return g_cuda.shared_mlp_w4a16(gate, up, down, y, x, S);
}

int coli_cuda_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales){
    if(!g_cuda.available){ return 0; }
    return g_cuda.tensor_update(tensor, weights, scales);
}



/* SYCL BACKEND */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_init            init;
    fn_shutdown        shutdown;
    fn_device_count    device_count;
    fn_device_at       device_at;
    fn_mem_info        mem_info;
    fn_stats           stats;
    fn_group_stats     group_stats;
    fn_expert_mlp      expert_mlp;
    fn_expert_group    expert_group;
    fn_attention_absorb attention_absorb;
    fn_tensor_upload   tensor_upload;
    fn_matmul          matmul;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;

    fn_attention_absorb_batch attention_absorb_batch;
    fn_attention_absorb_batch_dev attention_absorb_batch_dev;
    fn_attention_absorb_kvdev attention_absorb_kvdev;
    fn_attention_project_batch attention_project_batch;
    fn_attention_project_batch_dev attention_project_batch_dev;
    fn_attention_project_batch_dev_out attention_project_batch_dev_out;
    fn_pipe_add pipe_add;
    fn_pipe_alloc pipe_alloc;
    fn_pipe_copy2d pipe_copy2d;
    fn_pipe_download pipe_download;
    fn_pipe_free pipe_free;
    fn_pipe_gemm pipe_gemm;
    fn_pipe_peer_copy pipe_peer_copy;
    fn_pipe_rmsnorm pipe_rmsnorm;
    fn_pipe_rmsnorm_s pipe_rmsnorm_s;
    fn_pipe_rope pipe_rope;
    fn_pipe_rope_base pipe_rope_base;
    fn_pipe_rows_add pipe_rows_add;
    fn_pipe_scratch pipe_scratch;
    fn_pipe_silu_mul pipe_silu_mul;
    fn_pipe_sync pipe_sync;
    fn_pipe_upload pipe_upload;
    fn_shared_mlp_w4a16 shared_mlp_w4a16;
    fn_tensor_update tensor_update;
    void* (*alloc_mapped)(size_t bytes, void** device_ptr);
    void (*free_mapped)(void* host_ptr);
} g_sycl;
static int coli_sycl_load(void){
    if(g_sycl.loaded) return g_sycl.available;
    g_sycl.loaded = 1;

    /* Load coli_sycl.dll from the engine's OWN directory, by absolute path —
     * never a bare name. Resolving the path next to glm.exe and loading THAT
     * specific file with LOAD_WITH_ALTERED_SEARCH_PATH anchors both the DLL
     * and its dependency search to the trusted install directory instead of CWD. */
    char dllpath[MAX_PATH];
    HMODULE hm = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&coli_sycl_load, &hm);
    DWORD mn = GetModuleFileNameA(hm, dllpath, (DWORD)sizeof(dllpath));
    if(mn > 0 && mn < sizeof(dllpath)){
        char *slash = strrchr(dllpath, '\\');
        if(slash && (size_t)(slash + 1 - dllpath) + sizeof("coli_sycl.dll") <= sizeof(dllpath)){
            strcpy(slash + 1, "coli_sycl.dll");
            g_sycl.dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    if(!g_sycl.dll){
        /* fallback (GetModuleFileNameA almost never fails): search only
         * in the application directory and System32, NEVER the CWD. */
        g_sycl.dll = LoadLibraryExA("coli_sycl.dll", NULL,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if(!g_sycl.dll){
        fprintf(stderr, "[SYCL] coli_sycl.dll not found (or dependencies like oneAPI/SYCL runtime are missing from PATH); GPU tier disabled "
                        "(CPU path remains active).\n");
        return 0;
    }

    #define RESOLVE_SYCL(name, type) \
        /* GetProcAddress returns FARPROC (void(*)(void)); casting it to a   \
         * specific function-pointer type is the standard LoadLibrary idiom. \
         * -Wcast-function-type flags it but it is safe: the DLL exported     \
         * the symbol with extern "C" and the exact signature we expect. */   \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_sycl.name = (type)GetProcAddress(g_sycl.dll, "coli_sycl_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_sycl.name){ \
            fprintf(stderr, "[SYCL] coli_sycl.dll missing symbol coli_sycl_" #name "\n"); \
            FreeLibrary(g_sycl.dll); g_sycl.dll=NULL; return 0; }

    RESOLVE_SYCL(init,           fn_init)
    RESOLVE_SYCL(shutdown,       fn_shutdown)
    RESOLVE_SYCL(device_count,   fn_device_count)
    RESOLVE_SYCL(device_at,      fn_device_at)
    RESOLVE_SYCL(mem_info,       fn_mem_info)
    RESOLVE_SYCL(stats,          fn_stats)
    RESOLVE_SYCL(group_stats,    fn_group_stats)
    RESOLVE_SYCL(expert_mlp,     fn_expert_mlp)
    RESOLVE_SYCL(expert_group,   fn_expert_group)
    RESOLVE_SYCL(attention_absorb, fn_attention_absorb)
    RESOLVE_SYCL(tensor_upload,  fn_tensor_upload)
    RESOLVE_SYCL(matmul,         fn_matmul)
    RESOLVE_SYCL(tensor_free,    fn_tensor_free)
    RESOLVE_SYCL(tensor_bytes,   fn_tensor_bytes)
    RESOLVE_SYCL(tensor_device,  fn_tensor_device)

    RESOLVE_SYCL(attention_absorb_batch, fn_attention_absorb_batch)
    RESOLVE_SYCL(attention_absorb_batch_dev, fn_attention_absorb_batch_dev)
    RESOLVE_SYCL(attention_absorb_kvdev, fn_attention_absorb_kvdev)
    RESOLVE_SYCL(attention_project_batch, fn_attention_project_batch)
    RESOLVE_SYCL(attention_project_batch_dev, fn_attention_project_batch_dev)
    RESOLVE_SYCL(attention_project_batch_dev_out, fn_attention_project_batch_dev_out)
    RESOLVE_SYCL(pipe_add, fn_pipe_add)
    RESOLVE_SYCL(pipe_alloc, fn_pipe_alloc)
    RESOLVE_SYCL(pipe_copy2d, fn_pipe_copy2d)
    RESOLVE_SYCL(pipe_download, fn_pipe_download)
    RESOLVE_SYCL(pipe_free, fn_pipe_free)
    RESOLVE_SYCL(pipe_gemm, fn_pipe_gemm)
    RESOLVE_SYCL(pipe_peer_copy, fn_pipe_peer_copy)
    RESOLVE_SYCL(pipe_rmsnorm, fn_pipe_rmsnorm)
    RESOLVE_SYCL(pipe_rmsnorm_s, fn_pipe_rmsnorm_s)
    RESOLVE_SYCL(pipe_rope, fn_pipe_rope)
    RESOLVE_SYCL(pipe_rope_base, fn_pipe_rope_base)
    RESOLVE_SYCL(pipe_rows_add, fn_pipe_rows_add)
    RESOLVE_SYCL(pipe_scratch, fn_pipe_scratch)
    RESOLVE_SYCL(pipe_silu_mul, fn_pipe_silu_mul)
    RESOLVE_SYCL(pipe_sync, fn_pipe_sync)
    RESOLVE_SYCL(pipe_upload, fn_pipe_upload)
    RESOLVE_SYCL(shared_mlp_w4a16, fn_shared_mlp_w4a16)
    RESOLVE_SYCL(tensor_update, fn_tensor_update)
    #undef RESOLVE_SYCL

    g_sycl.available = 1;
    return 1;
}
int coli_sycl_init(const int *devices, int count){
    if(!coli_sycl_load()) return 0;
    return g_sycl.init(devices, count);
}

void coli_sycl_shutdown(void){
    if(g_sycl.available && g_sycl.shutdown) g_sycl.shutdown();
}

int coli_sycl_device_count(void){
    if(!g_sycl.available) return 0;
    return g_sycl.device_count();
}

int coli_sycl_device_at(int index){
    if(!g_sycl.available) return -1;
    return g_sycl.device_at(index);
}

int coli_sycl_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_sycl.available){ if(free_bytes)*free_bytes=0; if(total_bytes)*total_bytes=0; return 0; }
    return g_sycl.mem_info(device, free_bytes, total_bytes);
}

void coli_sycl_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    if(!g_sycl.available){ if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0; return; }
    g_sycl.stats(device, tensor_count, tensor_bytes);
}

void coli_sycl_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(!g_sycl.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_sycl.group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

int coli_sycl_expert_mlp(ColiSyclTensor *gate, ColiSyclTensor *up,
                         ColiSyclTensor *down, float *y, const float *x, int S){
    if(!g_sycl.available) return 0;
    return g_sycl.expert_mlp(gate, up, down, y, x, S);
}

int coli_sycl_expert_group(ColiSyclTensor *const *gates, ColiSyclTensor *const *ups,
                           ColiSyclTensor *const *downs, const int *rows, int count,
                           float *y, const float *x){
    if(!g_sycl.available) return 0;
    return g_sycl.expert_group(gates, ups, downs, rows, count, y, x);
}

int coli_sycl_attention_absorb(ColiSyclTensor *kv_b, float *ctx, const float *q,
                               const float *latent, const float *rope, int H, int Q,
                               int R, int V, int K, int T, float attention_scale){
    if(!g_sycl.available) return 0;
    return g_sycl.attention_absorb(kv_b, ctx, q, latent, rope, H, Q, R, V, K, T, attention_scale);
}

int coli_sycl_tensor_upload(ColiSyclTensor **tensor, const void *weights,
                            const float *scales, int fmt, int I, int O, int device){
    if(!g_sycl.available) return 0;
    return g_sycl.tensor_upload(tensor, weights, scales, fmt, I, O, device);
}

int coli_sycl_matmul(ColiSyclTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device){
    if(!g_sycl.available) return 0;
    return g_sycl.matmul(tensor, y, x, weights, scales, fmt, S, I, O, device);
}

void coli_sycl_tensor_free(ColiSyclTensor *tensor){
    if(g_sycl.available && g_sycl.tensor_free) g_sycl.tensor_free(tensor);
}

size_t coli_sycl_tensor_bytes(const ColiSyclTensor *tensor){
    if(!g_sycl.available) return 0;
    return g_sycl.tensor_bytes(tensor);
}

int coli_sycl_tensor_device(const ColiSyclTensor *tensor){
    if(!g_sycl.available) return -1;
    return g_sycl.tensor_device(tensor);
}

/* ---- #111 pipeline wrappers ---- */
void* coli_cuda_alloc_mapped(size_t bytes, void** device_ptr) {
    if(!g_cuda.dll) return NULL;
    return g_cuda.alloc_mapped(bytes, device_ptr);
}
void coli_cuda_free_mapped(void* host_ptr) {
    if(g_cuda.dll) g_cuda.free_mapped(host_ptr);
}



/* ---- #111 pipeline wrappers (see header for semantics) ---- */

int coli_sycl_attention_absorb_batch(ColiSyclTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale){
    if(!g_sycl.available){ return 0; }
    return g_sycl.attention_absorb_batch(kv_b, ctx, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_sycl_attention_absorb_batch_dev(ColiSyclTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_sycl.available){ return 0; }
    return g_sycl.attention_absorb_batch_dev(kv_b_shard, ctx_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_sycl_attention_absorb_kvdev(ColiSyclTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale){
    if(!g_sycl.available){ return 0; }
    return g_sycl.attention_absorb_kvdev(kv_b, ctx, q, latent_dev, rope_dev, H, Q, R, V, K, T, scale);
}

int coli_sycl_attention_project_batch(ColiSyclTensor *kv_b,ColiSyclTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale){
    if(!g_sycl.available){ return 0; }
    return g_sycl.attention_project_batch(kv_b, o_proj, out, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_sycl_attention_project_batch_dev(ColiSyclTensor *kv_b,ColiSyclTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_sycl.available){ return 0; }
    return g_sycl.attention_project_batch_dev(kv_b, o_proj, out, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_sycl_attention_project_batch_dev_out(ColiSyclTensor *kv_b,ColiSyclTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_sycl.available){ return 0; }
    return g_sycl.attention_project_batch_dev_out(kv_b, o_proj, out_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_sycl_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_add(device, x_dev, t_dev, n);
}

void * coli_sycl_pipe_alloc(int device,size_t bytes){
    if(!g_sycl.available){ return NULL; }
    return g_sycl.pipe_alloc(device, bytes);
}

int coli_sycl_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_copy2d(device, dst, dpitch, src, spitch, width, height);
}

int coli_sycl_pipe_download(int device,const void *src,void *dst,size_t bytes){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_download(device, src, dst, bytes);
}

void coli_sycl_pipe_free(int device,void *p){
    if(!g_sycl.available){ return; }
    g_sycl.pipe_free(device, p);
}

int coli_sycl_pipe_gemm(ColiSyclTensor *t,float *y_dev,const float *x_dev,int S){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_gemm(t, y_dev, x_dev, S);
}

int coli_sycl_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_peer_copy(dst_dev, dst, src_dev, src, bytes);
}

int coli_sycl_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_rmsnorm(device, y_dev, x_dev, w_dev, S, D, eps);
}

int coli_sycl_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_rmsnorm_s(device, y_dev, x_dev, w_dev, S, D, eps, xstride, ystride);
}

int coli_sycl_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_rope(device, v_dev, pos_dev, rows, stride, offset, R, heads, theta);
}

int coli_sycl_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_rope_base(device, v_dev, pos_base, rows, stride, offset, R, heads, theta);
}

int coli_sycl_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_rows_add(device, x_dev, partial_dev, rows_dev, nrows, D);
}

float * coli_sycl_pipe_scratch(int device,int slot,size_t bytes){
    if(!g_sycl.available){ return NULL; }
    return g_sycl.pipe_scratch(device, slot, bytes);
}

int coli_sycl_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_silu_mul(device, gate_dev, up_dev, n);
}

int coli_sycl_pipe_sync(int device){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_sync(device);
}

int coli_sycl_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    if(!g_sycl.available){ return 0; }
    return g_sycl.pipe_upload(device, dst, src, bytes);
}

int coli_sycl_shared_mlp_w4a16(ColiSyclTensor *gate, ColiSyclTensor *up, ColiSyclTensor *down, float *y, const float *x, int S){
    if(!g_sycl.available){ return 0; }
    return g_sycl.shared_mlp_w4a16(gate, up, down, y, x, S);
}

int coli_sycl_tensor_update(ColiSyclTensor *tensor, const void *weights, const float *scales){
    if(!g_sycl.available){ return 0; }
    return g_sycl.tensor_update(tensor, weights, scales);
}



/* VULKAN BACKEND */
static struct {
    int loaded;        /* 1 = load attempted (success or fail), 0 = not yet */
    int available;     /* 1 = DLL loaded and all symbols resolved */
    HMODULE dll;
    fn_init            init;
    fn_shutdown        shutdown;
    fn_device_count    device_count;
    fn_device_at       device_at;
    fn_mem_info        mem_info;
    fn_stats           stats;
    fn_group_stats     group_stats;
    fn_expert_mlp      expert_mlp;
    fn_expert_group    expert_group;
    fn_attention_absorb attention_absorb;
    fn_tensor_upload   tensor_upload;
    fn_matmul          matmul;
    fn_tensor_free     tensor_free;
    fn_tensor_bytes    tensor_bytes;
    fn_tensor_device   tensor_device;

    fn_attention_absorb_batch attention_absorb_batch;
    fn_attention_absorb_batch_dev attention_absorb_batch_dev;
    fn_attention_absorb_kvdev attention_absorb_kvdev;
    fn_attention_project_batch attention_project_batch;
    fn_attention_project_batch_dev attention_project_batch_dev;
    fn_attention_project_batch_dev_out attention_project_batch_dev_out;
    fn_pipe_add pipe_add;
    fn_pipe_alloc pipe_alloc;
    fn_pipe_copy2d pipe_copy2d;
    fn_pipe_download pipe_download;
    fn_pipe_free pipe_free;
    fn_pipe_gemm pipe_gemm;
    fn_pipe_peer_copy pipe_peer_copy;
    fn_pipe_rmsnorm pipe_rmsnorm;
    fn_pipe_rmsnorm_s pipe_rmsnorm_s;
    fn_pipe_rope pipe_rope;
    fn_pipe_rope_base pipe_rope_base;
    fn_pipe_rows_add pipe_rows_add;
    fn_pipe_scratch pipe_scratch;
    fn_pipe_silu_mul pipe_silu_mul;
    fn_pipe_sync pipe_sync;
    fn_pipe_upload pipe_upload;
    fn_shared_mlp_w4a16 shared_mlp_w4a16;
    fn_tensor_update tensor_update;
    void* (*alloc_mapped)(size_t bytes, void** device_ptr);
    void (*free_mapped)(void* host_ptr);
} g_vulkan;
static int coli_vulkan_load(void){
    if(g_vulkan.loaded) return g_vulkan.available;
    g_vulkan.loaded = 1;

    /* Load coli_vulkan.dll from the engine's OWN directory, by absolute path —
     * never a bare name. Resolving the path next to glm.exe and loading THAT
     * specific file with LOAD_WITH_ALTERED_SEARCH_PATH anchors both the DLL
     * and its dependency search to the trusted install directory instead of CWD. */
    char dllpath[MAX_PATH];
    HMODULE hm = NULL;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&coli_vulkan_load, &hm);
    DWORD mn = GetModuleFileNameA(hm, dllpath, (DWORD)sizeof(dllpath));
    if(mn > 0 && mn < sizeof(dllpath)){
        char *slash = strrchr(dllpath, '\\');
        if(slash && (size_t)(slash + 1 - dllpath) + sizeof("coli_vulkan.dll") <= sizeof(dllpath)){
            strcpy(slash + 1, "coli_vulkan.dll");
            g_vulkan.dll = LoadLibraryExA(dllpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        }
    }
    if(!g_vulkan.dll){
        /* fallback (GetModuleFileNameA almost never fails): search only
         * in the application directory and System32, NEVER the CWD. */
        g_vulkan.dll = LoadLibraryExA("coli_vulkan.dll", NULL,
            LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    }
    if(!g_vulkan.dll){
        fprintf(stderr, "[VULKAN] coli_vulkan.dll not found (or dependencies like Vulkan runtime are missing from PATH); GPU tier disabled "
                        "(CPU path remains active).\n");
        return 0;
    }

    #define RESOLVE_VULKAN(name, type) \
        /* GetProcAddress returns FARPROC (void(*)(void)); casting it to a   \
         * specific function-pointer type is the standard LoadLibrary idiom. \
         * -Wcast-function-type flags it but it is safe: the DLL exported     \
         * the symbol with extern "C" and the exact signature we expect. */   \
        _Pragma("GCC diagnostic push") \
        _Pragma("GCC diagnostic ignored \"-Wcast-function-type\"") \
        g_vulkan.name = (type)GetProcAddress(g_vulkan.dll, "coli_vulkan_" #name); \
        _Pragma("GCC diagnostic pop") \
        if(!g_vulkan.name){ \
            fprintf(stderr, "[VULKAN] coli_vulkan.dll missing symbol coli_vulkan_" #name "\n"); \
            FreeLibrary(g_vulkan.dll); g_vulkan.dll=NULL; return 0; }

    RESOLVE_VULKAN(init,           fn_init)
    RESOLVE_VULKAN(shutdown,       fn_shutdown)
    RESOLVE_VULKAN(device_count,   fn_device_count)
    RESOLVE_VULKAN(device_at,      fn_device_at)
    RESOLVE_VULKAN(mem_info,       fn_mem_info)
    RESOLVE_VULKAN(stats,          fn_stats)
    RESOLVE_VULKAN(group_stats,    fn_group_stats)
    RESOLVE_VULKAN(expert_mlp,     fn_expert_mlp)
    RESOLVE_VULKAN(expert_group,   fn_expert_group)
    RESOLVE_VULKAN(attention_absorb, fn_attention_absorb)
    RESOLVE_VULKAN(tensor_upload,  fn_tensor_upload)
    RESOLVE_VULKAN(matmul,         fn_matmul)
    RESOLVE_VULKAN(tensor_free,    fn_tensor_free)
    RESOLVE_VULKAN(tensor_bytes,   fn_tensor_bytes)
    RESOLVE_VULKAN(tensor_device,  fn_tensor_device)

    RESOLVE_VULKAN(attention_absorb_batch, fn_attention_absorb_batch)
    RESOLVE_VULKAN(attention_absorb_batch_dev, fn_attention_absorb_batch_dev)
    RESOLVE_VULKAN(attention_absorb_kvdev, fn_attention_absorb_kvdev)
    RESOLVE_VULKAN(attention_project_batch, fn_attention_project_batch)
    RESOLVE_VULKAN(attention_project_batch_dev, fn_attention_project_batch_dev)
    RESOLVE_VULKAN(attention_project_batch_dev_out, fn_attention_project_batch_dev_out)
    RESOLVE_VULKAN(pipe_add, fn_pipe_add)
    RESOLVE_VULKAN(pipe_alloc, fn_pipe_alloc)
    RESOLVE_VULKAN(pipe_copy2d, fn_pipe_copy2d)
    RESOLVE_VULKAN(pipe_download, fn_pipe_download)
    RESOLVE_VULKAN(pipe_free, fn_pipe_free)
    RESOLVE_VULKAN(pipe_gemm, fn_pipe_gemm)
    RESOLVE_VULKAN(pipe_peer_copy, fn_pipe_peer_copy)
    RESOLVE_VULKAN(pipe_rmsnorm, fn_pipe_rmsnorm)
    RESOLVE_VULKAN(pipe_rmsnorm_s, fn_pipe_rmsnorm_s)
    RESOLVE_VULKAN(pipe_rope, fn_pipe_rope)
    RESOLVE_VULKAN(pipe_rope_base, fn_pipe_rope_base)
    RESOLVE_VULKAN(pipe_rows_add, fn_pipe_rows_add)
    RESOLVE_VULKAN(pipe_scratch, fn_pipe_scratch)
    RESOLVE_VULKAN(pipe_silu_mul, fn_pipe_silu_mul)
    RESOLVE_VULKAN(pipe_sync, fn_pipe_sync)
    RESOLVE_VULKAN(pipe_upload, fn_pipe_upload)
    RESOLVE_VULKAN(shared_mlp_w4a16, fn_shared_mlp_w4a16)
    RESOLVE_VULKAN(tensor_update, fn_tensor_update)
    #undef RESOLVE_VULKAN

    g_vulkan.available = 1;
    return 1;
}
int coli_vulkan_init(const int *devices, int count){
    if(!coli_vulkan_load()) return 0;
    return g_vulkan.init(devices, count);
}

void coli_vulkan_shutdown(void){
    if(g_vulkan.available && g_vulkan.shutdown) g_vulkan.shutdown();
}

int coli_vulkan_device_count(void){
    if(!g_vulkan.available) return 0;
    return g_vulkan.device_count();
}

int coli_vulkan_device_at(int index){
    if(!g_vulkan.available) return -1;
    return g_vulkan.device_at(index);
}

int coli_vulkan_mem_info(int device, size_t *free_bytes, size_t *total_bytes){
    if(!g_vulkan.available){ if(free_bytes)*free_bytes=0; if(total_bytes)*total_bytes=0; return 0; }
    return g_vulkan.mem_info(device, free_bytes, total_bytes);
}

void coli_vulkan_stats(int device, size_t *tensor_count, size_t *tensor_bytes){
    if(!g_vulkan.available){ if(tensor_count)*tensor_count=0; if(tensor_bytes)*tensor_bytes=0; return; }
    g_vulkan.stats(device, tensor_count, tensor_bytes);
}

void coli_vulkan_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms){
    if(!g_vulkan.available){
        if(calls)*calls=0; if(experts)*experts=0; if(rows)*rows=0;
        if(h2d_ms)*h2d_ms=0; if(kernel_ms)*kernel_ms=0; if(d2h_ms)*d2h_ms=0;
        return;
    }
    g_vulkan.group_stats(calls, experts, rows, h2d_ms, kernel_ms, d2h_ms);
}

int coli_vulkan_expert_mlp(ColiVulkanTensor *gate, ColiVulkanTensor *up,
                         ColiVulkanTensor *down, float *y, const float *x, int S){
    if(!g_vulkan.available) return 0;
    return g_vulkan.expert_mlp(gate, up, down, y, x, S);
}

int coli_vulkan_expert_group(ColiVulkanTensor *const *gates, ColiVulkanTensor *const *ups,
                           ColiVulkanTensor *const *downs, const int *rows, int count,
                           float *y, const float *x){
    if(!g_vulkan.available) return 0;
    return g_vulkan.expert_group(gates, ups, downs, rows, count, y, x);
}

int coli_vulkan_attention_absorb(ColiVulkanTensor *kv_b, float *ctx, const float *q,
                               const float *latent, const float *rope, int H, int Q,
                               int R, int V, int K, int T, float attention_scale){
    if(!g_vulkan.available) return 0;
    return g_vulkan.attention_absorb(kv_b, ctx, q, latent, rope, H, Q, R, V, K, T, attention_scale);
}

int coli_vulkan_tensor_upload(ColiVulkanTensor **tensor, const void *weights,
                            const float *scales, int fmt, int I, int O, int device){
    if(!g_vulkan.available) return 0;
    return g_vulkan.tensor_upload(tensor, weights, scales, fmt, I, O, device);
}

int coli_vulkan_matmul(ColiVulkanTensor **tensor, float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device){
    if(!g_vulkan.available) return 0;
    return g_vulkan.matmul(tensor, y, x, weights, scales, fmt, S, I, O, device);
}

void coli_vulkan_tensor_free(ColiVulkanTensor *tensor){
    if(g_vulkan.available && g_vulkan.tensor_free) g_vulkan.tensor_free(tensor);
}

size_t coli_vulkan_tensor_bytes(const ColiVulkanTensor *tensor){
    if(!g_vulkan.available) return 0;
    return g_vulkan.tensor_bytes(tensor);
}

int coli_vulkan_tensor_device(const ColiVulkanTensor *tensor){
    if(!g_vulkan.available) return -1;
    return g_vulkan.tensor_device(tensor);
}

/* ---- #111 pipeline wrappers ---- */
void* coli_cuda_alloc_mapped(size_t bytes, void** device_ptr) {
    if(!g_cuda.dll) return NULL;
    return g_cuda.alloc_mapped(bytes, device_ptr);
}
void coli_cuda_free_mapped(void* host_ptr) {
    if(g_cuda.dll) g_cuda.free_mapped(host_ptr);
}



/* ---- #111 pipeline wrappers (see header for semantics) ---- */

int coli_vulkan_attention_absorb_batch(ColiVulkanTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.attention_absorb_batch(kv_b, ctx, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_vulkan_attention_absorb_batch_dev(ColiVulkanTensor *kv_b_shard,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.attention_absorb_batch_dev(kv_b_shard, ctx_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_vulkan_attention_absorb_kvdev(ColiVulkanTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.attention_absorb_kvdev(kv_b, ctx, q, latent_dev, rope_dev, H, Q, R, V, K, T, scale);
}

int coli_vulkan_attention_project_batch(ColiVulkanTensor *kv_b,ColiVulkanTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.attention_project_batch(kv_b, o_proj, out, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

int coli_vulkan_attention_project_batch_dev(ColiVulkanTensor *kv_b,ColiVulkanTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.attention_project_batch_dev(kv_b, o_proj, out, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_vulkan_attention_project_batch_dev_out(ColiVulkanTensor *kv_b,ColiVulkanTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.attention_project_batch_dev_out(kv_b, o_proj, out_dev, q_dev, latent_dev, rope_dev, S, H, Q, R, V, K, T, scale);
}

int coli_vulkan_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_add(device, x_dev, t_dev, n);
}

void * coli_vulkan_pipe_alloc(int device,size_t bytes){
    if(!g_vulkan.available){ return NULL; }
    return g_vulkan.pipe_alloc(device, bytes);
}

int coli_vulkan_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_copy2d(device, dst, dpitch, src, spitch, width, height);
}

int coli_vulkan_pipe_download(int device,const void *src,void *dst,size_t bytes){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_download(device, src, dst, bytes);
}

void coli_vulkan_pipe_free(int device,void *p){
    if(!g_vulkan.available){ return; }
    g_vulkan.pipe_free(device, p);
}

int coli_vulkan_pipe_gemm(ColiVulkanTensor *t,float *y_dev,const float *x_dev,int S){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_gemm(t, y_dev, x_dev, S);
}

int coli_vulkan_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_peer_copy(dst_dev, dst, src_dev, src, bytes);
}

int coli_vulkan_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_rmsnorm(device, y_dev, x_dev, w_dev, S, D, eps);
}

int coli_vulkan_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_rmsnorm_s(device, y_dev, x_dev, w_dev, S, D, eps, xstride, ystride);
}

int coli_vulkan_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_rope(device, v_dev, pos_dev, rows, stride, offset, R, heads, theta);
}

int coli_vulkan_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_rope_base(device, v_dev, pos_base, rows, stride, offset, R, heads, theta);
}

int coli_vulkan_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_rows_add(device, x_dev, partial_dev, rows_dev, nrows, D);
}

float * coli_vulkan_pipe_scratch(int device,int slot,size_t bytes){
    if(!g_vulkan.available){ return NULL; }
    return g_vulkan.pipe_scratch(device, slot, bytes);
}

int coli_vulkan_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_silu_mul(device, gate_dev, up_dev, n);
}

int coli_vulkan_pipe_sync(int device){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_sync(device);
}

int coli_vulkan_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.pipe_upload(device, dst, src, bytes);
}

int coli_vulkan_shared_mlp_w4a16(ColiVulkanTensor *gate, ColiVulkanTensor *up, ColiVulkanTensor *down, float *y, const float *x, int S){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.shared_mlp_w4a16(gate, up, down, y, x, S);
}

int coli_vulkan_tensor_update(ColiVulkanTensor *tensor, const void *weights, const float *scales){
    if(!g_vulkan.available){ return 0; }
    return g_vulkan.tensor_update(tensor, weights, scales);
}


void* coli_sycl_alloc_mapped(size_t bytes, void** device_ptr) {
    if(!g_sycl.dll) return NULL;
    return g_sycl.alloc_mapped(bytes, device_ptr);
}
void coli_sycl_free_mapped(void* host_ptr) {
    if(g_sycl.dll) g_sycl.free_mapped(host_ptr);
}
void* coli_vulkan_alloc_mapped(size_t bytes, void** device_ptr) {
    if(!g_vulkan.dll) return NULL;
    return g_vulkan.alloc_mapped(bytes, device_ptr);
}
void coli_vulkan_free_mapped(void* host_ptr) {
    if(g_vulkan.dll) g_vulkan.free_mapped(host_ptr);
}
#endif /* _WIN32 */
