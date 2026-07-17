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
    struct event {
        void wait() const {}
    };
    struct queue {
        queue() {}
        template<typename Property> queue(Property) {}
        void wait() const {}
        template<typename Range, typename Func>
        event parallel_for(Range r, Func f) { return event{}; }
        template<typename Func>
        event submit(Func f) { return event{}; }
        void* get_device() { return nullptr; }
        event memcpy(void* dst, const void* src, size_t size) { return event{}; }
    };
    namespace info {
        namespace device {
            struct ext_intel_free_memory { static const int value = 1; };
            struct global_mem_size { static const int value = 2; };
        }
    }
    namespace property { namespace queue { struct in_order {}; } }
    template<int dim> struct range { 
        template<typename... Args> range(Args... args) {}
    };
    template<int dim> struct id { size_t operator[](int) const { return 0; } };
    
    struct group {
        template<typename T> void barrier(T) const {}
        size_t get_local_linear_range() const { return 1; }
        size_t get_local_linear_id() const { return 0; }
    };
    
    struct nd_range_t {
        template<int dim> nd_range_t(range<dim> global, range<dim> local) {}
    };
    template<int dim> struct nd_range : nd_range_t { using nd_range_t::nd_range_t; };
    
    template<int dim> struct nd_item {
        group get_group() const { return group{}; }
        size_t get_local_id(int) const { return 0; }
        size_t get_group(int) const { return 0; }
        template<typename F> void barrier(F) const {}
    };
    
    struct handler {
        template<int dim, typename Func>
        void parallel_for(nd_range<dim> r, Func f) {}
    };
    
    namespace access { enum class decorated { no, yes }; enum class fence_space { local_space }; }

    template<typename T, int dim=1> struct local_accessor {
        template<typename H> local_accessor(range<dim>, H&) {}
        T& operator[](size_t) { static T dummy; return dummy; }

        template<access::decorated D = access::decorated::no>
        struct MultiPtr {
            T* get() const { return nullptr; }
        };

        template<access::decorated D = access::decorated::no>
        MultiPtr<D> get_multi_ptr() const { return {}; }
    };
    
    
    template<typename T=void> struct plus { 
        T operator()(T a, T b) const { return a + b; } 
    }; 
    template<> struct plus<void> { 
        template<typename T> T operator()(T a, T b) const { return a + b; } 
    };
    
    template<typename T=void> struct maximum { 
        T operator()(T a, T b) const { return a > b ? a : b; } 
    }; 
    template<> struct maximum<void> { 
        template<typename T> T operator()(T a, T b) const { return a > b ? a : b; } 
    };
    
    template<typename Grp, typename T, typename Op> 
    T reduce_over_group(Grp g, T x, Op op) { return x; }

    float exp(float x) { return std::exp(x); }
    float sqrt(float x) { return std::sqrt(x); }
    float pow(float base, float ex) { return std::pow(base, ex); }
    float cos(float x) { return std::cos(x); }
    float sin(float x) { return std::sin(x); }
    template<typename T> T* malloc_device(size_t sz, queue& q) { return (T*)malloc(sz * sizeof(T)); }
    template<typename T> T* malloc_host(size_t sz, queue& q) { return (T*)malloc(sz * sizeof(T)); }
    void free(void* ptr, queue& q) { ::free(ptr); }
}
#endif

static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    if (fmt == 2) {
        uint8_t v = base[i >> 1];
        int n = (i & 1) ? (v >> 4) : (v & 15);
        return static_cast<float>(n & 8 ? n - 16 : n);
    }
    uint8_t v = base[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

// ---------------------------------------------------------------------------------
// Context and State Management
// ---------------------------------------------------------------------------------

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

struct DeviceContext {
    int device;
    sycl::queue* q;
    size_t tensor_count;
    size_t tensor_bytes;

    // Scratch spaces
    float* pipe_buf[24];
    size_t pipe_cap[24];

    // Execution buffers (similar to CUDA backend)
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    float *aq, *al, *ar, *ac;
    size_t aq_cap, al_cap, ar_cap, ac_cap;
};

static DeviceContext g_ctx[COLI_SYCL_MAX_DEVICES];
static int g_nctx = 0;

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) {
        if (g_ctx[i].device == device) return &g_ctx[i];
    }
    return nullptr;
}

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

// ---------------------------------------------------------------------------------
// Initialization & Information
// ---------------------------------------------------------------------------------

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_init(const int *devices, int count) {
    if (!devices || count < 1 || count > COLI_SYCL_MAX_DEVICES) return 0;

    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        DeviceContext *ctx = &g_ctx[g_nctx++];
        memset(ctx, 0, sizeof(DeviceContext));
        ctx->device = devices[i];
        ctx->q = new sycl::queue(sycl::property::queue::in_order()); // Must be in-order to prevent data races
    }
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT void coli_sycl_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (ctx->q) {
            for (int s = 0; s < 24; s++) {
                if (ctx->pipe_buf[s]) sycl::free(ctx->pipe_buf[s], *ctx->q);
            }
            if (ctx->x) sycl::free(ctx->x, *ctx->q);
            if (ctx->y) sycl::free(ctx->y, *ctx->q);
            if (ctx->gate) sycl::free(ctx->gate, *ctx->q);
            if (ctx->up) sycl::free(ctx->up, *ctx->q);
            if (ctx->aq) sycl::free(ctx->aq, *ctx->q);
            if (ctx->al) sycl::free(ctx->al, *ctx->q);
            if (ctx->ar) sycl::free(ctx->ar, *ctx->q);
            if (ctx->ac) sycl::free(ctx->ac, *ctx->q);

            delete ctx->q;
            ctx->q = nullptr;
        }
    }
    g_nctx = 0;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_device_count(void) { return g_nctx; }
extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_device_at(int index) {
    if(index < 0 || index >= g_nctx) return -1;
    return g_ctx[index].device;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || !free_bytes || !total_bytes) return 0;

    // In actual SYCL, free memory might require platform specific extensions:
    // e.g., ctx->q->get_device().get_info<sycl::ext::intel::info::device::free_memory>();
    // For universal compatibility, if it fails or is unavailable, we look at CUDA_EXPERT_GB.

#if defined(_MSC_VER)
    char* env_gb = nullptr;
    size_t env_sz = 0;
    _dupenv_s(&env_gb, &env_sz, "CUDA_EXPERT_GB");
#else
    const char* env_gb = std::getenv("CUDA_EXPERT_GB");
#endif
    if (env_gb) {
        double val = std::atof(env_gb);
#if defined(_MSC_VER)
        free(env_gb);
#endif
        if (val > 0) {
            size_t gb = (size_t)(val * 1024 * 1024 * 1024);
            *free_bytes = gb;
            *total_bytes = gb;
            return 1;
        }
    }

    // Fallback: Report a generic large memory footprint to ensure it attempts VRAM allocation
    *free_bytes = 24ULL * 1024 * 1024 * 1024;
    *total_bytes = 24ULL * 1024 * 1024 * 1024;
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT void coli_sycl_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

uint64_t g_group_calls=0, g_group_experts=0, g_group_rows=0;
double g_group_h2d_ms=0, g_group_kernel_ms=0, g_group_d2h_ms=0;

extern "C" COLI_SYCL_DLLEXPORT void coli_sycl_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows, double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

// ---------------------------------------------------------------------------------
// Memory Uploads and Management
// ---------------------------------------------------------------------------------

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_tensor_upload(ColiCudaTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx) return 0;

    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }

    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;

    t->weights = sycl::malloc_device<uint8_t>(t->weight_bytes, *ctx->q);
    if (!t->weights) { free(t); return 0; }

    ctx->q->memcpy(t->weights, weights, t->weight_bytes).wait();

    if (fmt == 2) {
        uint8_t *w = (uint8_t*)t->weights;
        size_t bytes = t->weight_bytes;
        ctx->q->parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> i) {
            w[i[0]] ^= 0x88;
        }).wait();
    }

    if (fmt) {
        t->scales = sycl::malloc_device<float>(O, *ctx->q);
        if (!t->scales) { sycl::free(t->weights, *ctx->q); free(t); return 0; }
        ctx->q->memcpy(t->scales, scales, (size_t)O * sizeof(float)).wait();
    }

    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);

    *tensor = t;
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT void coli_sycl_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx && tensor->tracked) {
        ctx->tensor_count--;
        ctx->tensor_bytes -= tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
    }
    if (tensor->weights && ctx) sycl::free(tensor->weights, *ctx->q);
    if (tensor->scales && ctx) sycl::free(tensor->scales, *ctx->q);
    free(tensor);
}

extern "C" COLI_SYCL_DLLEXPORT size_t coli_sycl_tensor_bytes(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    return tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_tensor_device(const ColiCudaTensor *tensor) {
    if (!tensor) return -1;
    return tensor->device;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_tensor_update(ColiCudaTensor *tensor, const void *weights, const float *scales) {
    if (!tensor || !weights || (tensor->fmt && !scales)) return 0;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (!ctx) return 0;

    ctx->q->memcpy(tensor->weights, weights, tensor->weight_bytes).wait();

    if (tensor->fmt == 2) {
        uint8_t *w = (uint8_t*)tensor->weights;
        size_t bytes = tensor->weight_bytes;
        ctx->q->parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> i) {
            w[i[0]] ^= 0x88;
        }).wait();
    }

    if (tensor->fmt) {
        ctx->q->memcpy(tensor->scales, scales, (size_t)tensor->O * sizeof(float)).wait();
    }
    return 1;
}

// ---------------------------------------------------------------------------------
// Pipeline Scratch and Buffers
// ---------------------------------------------------------------------------------

static int reserve_buf(DeviceContext *ctx, float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) sycl::free(*ptr, *ctx->q);
    *ptr = nullptr;
    *cap = 0;
    *ptr = sycl::malloc_device<float>(bytes / sizeof(float), *ctx->q);
    if (!*ptr) return 0;
    *cap = bytes;
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT float *coli_sycl_pipe_scratch(int device,int slot,size_t bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || slot < 0 || slot >= 24) return nullptr;
    if (!reserve_buf(ctx, &ctx->pipe_buf[slot], &ctx->pipe_cap[slot], bytes)) return nullptr;
    return ctx->pipe_buf[slot];
}

extern "C" COLI_SYCL_DLLEXPORT void *coli_sycl_pipe_alloc(int device,size_t bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx) return nullptr;
    return sycl::malloc_device<uint8_t>(bytes, *ctx->q);
}

extern "C" COLI_SYCL_DLLEXPORT void coli_sycl_pipe_free(int device,void *p) {
    DeviceContext *ctx = find_ctx(device);
    if (ctx && p) sycl::free(p, *ctx->q);
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_upload(int device,void *dst,const void *src,size_t bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx) return 0;
    ctx->q->memcpy(dst, src, bytes).wait();
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_download(int device,const void *src,void *dst,size_t bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx) return 0;
    ctx->q->memcpy(dst, src, bytes).wait();
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_copy2d(int device,float *dst,int dpitch,const float *src, int spitch,int width,int height) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx) return 0;
    // Basic fallback for 2D copy
    for (int h = 0; h < height; ++h) {
        ctx->q->memcpy(dst + h * dpitch, src + h * spitch, width * sizeof(float)).wait();
    }
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_peer_copy(int dst_dev,float *dst,int src_dev, const float *src,size_t bytes) {
    DeviceContext *ctx = find_ctx(dst_dev);
    if (!ctx) return 0;
    // Assuming unified memory or driver routing handles D2D
    ctx->q->memcpy(dst, src, bytes).wait();
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_sync(int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx) return 0;
    ctx->q->wait();
    return 1;
}


// ---------------------------------------------------------------------------------
// Element-wise Compute Primitives
// ---------------------------------------------------------------------------------

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rmsnorm(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || S < 1 || D < 1) return 0;

    ctx->q->parallel_for(sycl::range<1>(S), [=](sycl::id<1> s_id) {
        int s = s_id[0];
        const float* xs = x_dev + s * D;
        float* ys = y_dev + s * D;

        float sum = 0.0f;
        for (int i = 0; i < D; ++i) {
            sum += xs[i] * xs[i];
        }

        float inv_rms = 1.0f / sycl::sqrt(sum / D + eps);
        for (int i = 0; i < D; ++i) {
            ys[i] = xs[i] * inv_rms * w_dev[i];
        }
    });
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev, const float *w_dev,int S,int D,float eps, int xstride,int ystride) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || S < 1 || D < 1) return 0;

    ctx->q->parallel_for(sycl::range<1>(S), [=](sycl::id<1> s_id) {
        int s = s_id[0];
        const float* xs = x_dev + s * xstride;
        float* ys = y_dev + s * ystride;

        float sum = 0.0f;
        for (int i = 0; i < D; ++i) {
            sum += xs[i] * xs[i];
        }

        float inv_rms = 1.0f / sycl::sqrt(sum / D + eps);
        for (int i = 0; i < D; ++i) {
            ys[i] = xs[i] * inv_rms * w_dev[i];
        }
    });
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows, int stride,int offset,int R,int heads,float theta) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || rows < 1 || R < 2 || heads < 1) return 0;

    ctx->q->parallel_for(sycl::range<1>(rows * heads * (R / 2)), [=](sycl::id<1> idx) {
        int r = idx[0];
        int d = (r % (R / 2)) * 2;
        int h = (r / (R / 2)) % heads;
        int row = r / (heads * (R / 2));

        int pos = pos_dev ? pos_dev[row] : 0;
        float freq = pos * sycl::pow(theta, -((float)d / R));
        float cos_val = sycl::cos(freq);
        float sin_val = sycl::sin(freq);

        float* v = v_dev + row * stride + offset + h * R + d;
        float v0 = v[0], v1 = v[1];
        v[0] = v0 * cos_val - v1 * sin_val;
        v[1] = v0 * sin_val + v1 * cos_val;
    });
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rope_base(int device,float *v_dev,int pos_base,int rows, int stride,int offset,int R,int heads,float theta) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || rows < 1 || R < 2 || heads < 1) return 0;

    ctx->q->parallel_for(sycl::range<1>(rows * heads * (R / 2)), [=](sycl::id<1> idx) {
        int r = idx[0];
        int d = (r % (R / 2)) * 2;
        int h = (r / (R / 2)) % heads;
        int row = r / (heads * (R / 2));

        int pos = pos_base + row;
        float freq = pos * sycl::pow(theta, -((float)d / R));
        float cos_val = sycl::cos(freq);
        float sin_val = sycl::sin(freq);

        float* v = v_dev + row * stride + offset + h * R + d;
        float v0 = v[0], v1 = v[1];
        v[0] = v0 * cos_val - v1 * sin_val;
        v[1] = v0 * sin_val + v1 * cos_val;
    });
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || !n) return 0;

    ctx->q->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        float x = gate_dev[i[0]];
        gate_dev[i[0]] = (x / (1.0f + sycl::exp(-x))) * up_dev[i[0]];
    });
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_add(int device,float *x_dev,const float *t_dev,size_t n) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || !n) return 0;
    ctx->q->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        x_dev[i[0]] += t_dev[i[0]];
    });
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_rows_add(int device,float *x_dev,const float *partial_dev, const int *rows_dev,int nrows,int D) {
    DeviceContext *ctx = find_ctx(device);
    if (!ctx || nrows < 1 || D < 1) return 0;

    ctx->q->parallel_for(sycl::range<2>(nrows, D), [=](sycl::id<2> idx) {
        int r = idx[0];
        int d = idx[1];
        int target_row = rows_dev[r];
        x_dev[target_row * D + d] += partial_dev[r * D + d];
    });
    return 1;
}


// ---------------------------------------------------------------------------------
// GEMM and MLPs
// ---------------------------------------------------------------------------------

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_matmul(ColiCudaTensor **tensor, float *y, const float *x, const void *weights, const float *scales, int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_sycl_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!ctx) return 0;

    // Allocate device memory for x and y
    size_t xb = S * I * sizeof(float);
    size_t yb = S * O * sizeof(float);
    if (!reserve_buf(ctx, &ctx->x, &ctx->x_cap, xb)) return 0;
    if (!reserve_buf(ctx, &ctx->y, &ctx->y_cap, yb)) return 0;

    ctx->q->memcpy(ctx->x, x, xb).wait();

    float* x_dev = ctx->x;
    float* y_dev = ctx->y;
    const void* w_dev = t->weights;
    const float* s_dev = t->scales;
    size_t rb = row_bytes(fmt, I);

    ctx->q->parallel_for(sycl::range<2>(S, O), [=](sycl::id<2> idx) {
        int s = idx[0];
        int o = idx[1];

        float sum = 0.0f;
        const float* xs = x_dev + s * I;

        for (int i = 0; i < I; ++i) {
            sum += xs[i] * weight_at(w_dev, fmt, o * rb, i);
        }

        y_dev[s * O + o] = sum * (fmt ? s_dev[o] : 1.0f);
    }).wait(); // Adding .wait() to simplify stream/pipeline logic right now

    ctx->q->memcpy(y, ctx->y, yb).wait();
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S) {
    if(!t || S < 1) return 0;
    DeviceContext *ctx = find_ctx(t->device);
    if (!ctx) return 0;

    int O = t->O;
    int I = t->I;
    int fmt = t->fmt;
    const void* w_dev = t->weights;
    const float* s_dev = t->scales;
    size_t rb = row_bytes(fmt, I);

    ctx->q->parallel_for(sycl::range<2>(S, O), [=](sycl::id<2> idx) {
        int s = idx[0];
        int o = idx[1];

        float sum = 0.0f;
        const float* xs = x_dev + s * I;

        for (int i = 0; i < I; ++i) {
            sum += xs[i] * weight_at(w_dev, fmt, o * rb, i);
        }

        y_dev[s * O + o] = sum * (fmt ? s_dev[o] : 1.0f);
    });

    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S) {
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;

    DeviceContext *ctx = find_ctx(gate->device);
    if (!ctx) return 0;

    int D = gate->I, I = gate->O;
    size_t xb = S * D * sizeof(float);
    size_t ib = S * I * sizeof(float);
    size_t yb = S * D * sizeof(float);

    if (!reserve_buf(ctx, &ctx->x, &ctx->x_cap, xb) ||
        !reserve_buf(ctx, &ctx->y, &ctx->y_cap, yb) ||
        !reserve_buf(ctx, &ctx->gate, &ctx->gate_cap, ib) ||
        !reserve_buf(ctx, &ctx->up, &ctx->up_cap, ib)) return 0;

    ctx->q->memcpy(ctx->x, x, xb).wait();

    coli_sycl_pipe_gemm(gate, ctx->gate, ctx->x, S);
    coli_sycl_pipe_gemm(up, ctx->up, ctx->x, S);
    coli_sycl_pipe_silu_mul(gate->device, ctx->gate, ctx->up, S * I);
    coli_sycl_pipe_gemm(down, ctx->y, ctx->gate, S);

    ctx->q->memcpy(y, ctx->y, yb).wait();
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up, ColiCudaTensor *down, float *y, const float *x, int S) {
    // Fall back to the generic pipeline
    return coli_sycl_expert_mlp(gate, up, down, y, x, S);
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_expert_group(ColiCudaTensor *const *gates, ColiCudaTensor *const *ups, ColiCudaTensor *const *downs, const int *rows, int count, float *y, const float *x) {
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first = gates[0];
    if (!first) return 0;

    int device = first->device;
    int D = first->I;
    int I = first->O;

    DeviceContext *ctx = find_ctx(device);
    if (!ctx) return 0;

    int total_rows = 0;
    for (int i = 0; i < count; ++i) total_rows += rows[i];

    size_t xb = total_rows * D * sizeof(float);
    size_t ib = total_rows * I * sizeof(float);
    size_t yb = total_rows * D * sizeof(float);

    if (!reserve_buf(ctx, &ctx->x, &ctx->x_cap, xb) ||
        !reserve_buf(ctx, &ctx->y, &ctx->y_cap, yb) ||
        !reserve_buf(ctx, &ctx->gate, &ctx->gate_cap, ib) ||
        !reserve_buf(ctx, &ctx->up, &ctx->up_cap, ib)) return 0;

    ctx->q->memcpy(ctx->x, x, xb).wait();

    int offset = 0;
    for (int c = 0; c < count; ++c) {
        int r = rows[c];
        if (r == 0) continue;

        ColiCudaTensor* g = gates[c];
        ColiCudaTensor* u = ups[c];
        ColiCudaTensor* d = downs[c];

        float* x_ptr = ctx->x + offset * D;
        float* g_ptr = ctx->gate + offset * I;
        float* u_ptr = ctx->up + offset * I;
        float* y_ptr = ctx->y + offset * D;

        // Use generic pipe gemm on sub-tensors
        coli_sycl_pipe_gemm(g, g_ptr, x_ptr, r);
        coli_sycl_pipe_gemm(u, u_ptr, x_ptr, r);

        // Wait because silu_mul uses the full array conceptually but we can just use the pointer
        // Wait not strictly required if everything is in the same in-order queue, but safe

        ctx->q->parallel_for(sycl::range<1>(r * I), [=](sycl::id<1> idx) {
            float x_val = g_ptr[idx[0]];
            g_ptr[idx[0]] = (x_val / (1.0f + sycl::exp(-x_val))) * u_ptr[idx[0]];
        });

        coli_sycl_pipe_gemm(d, y_ptr, g_ptr, r);

        offset += r;
    }

    ctx->q->memcpy(y, ctx->y, yb).wait();
    return 1;
}


// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// Attention Primitives
// ---------------------------------------------------------------------------------

static void attention_absorb_batch_kernel(sycl::queue &q, float *ctx, const float *q_dev,
        const float *latent, const float *rope, const void *weights, const float *wscale,
        int fmt, int S, int H, int Q, int R, int V, int K, int T, float scale) {
    size_t rb = row_bytes(fmt, K);
    
    // We launch H * S work-groups.
    sycl::range<2> global_range(S, H * 256);
    sycl::range<2> local_range(1, 256);
    
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sm(sycl::range<1>(2 * K + T + 256), cgh);
        
        cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
            int s = item.get_group(0);
            int h = item.get_group(1);
            int tid = item.get_local_id(1);
            int nt = T - S + s + 1;
            int rbase = h * (Q + V);
            
            if (s >= S || nt < 1) return;
            
            float* qa = sm.template get_multi_ptr<sycl::access::decorated::no>().get();
            float* cl = qa + K;
            float* scores = cl + K;
            
            const float *qs = q_dev + ((size_t)s * H + h) * (Q + R);
            
            for(int k = tid; k < K; k += 256) {
                float a = 0;
                for(int d = 0; d < Q; d++) {
                    a += qs[d] * weight_at(weights, fmt, (size_t)(rbase + d) * rb, k) * (fmt ? wscale[rbase + d] : 1.f);
                }
                qa[k] = a;
            }
            item.barrier(sycl::access::fence_space::local_space);
            
            for(int t = tid; t < nt; t += 256) {
                float a = 0;
                const float *lt = latent + (size_t)t * K;
                const float *rt = rope + (size_t)t * R;
                for(int k = 0; k < K; k++) a += qa[k] * lt[k];
                for(int d = 0; d < R; d++) a += qs[Q + d] * rt[d];
                scores[t] = a * scale;
            }
            item.barrier(sycl::access::fence_space::local_space);
            
            float local_max = -3.402823466e+38F;
            for(int t = tid; t < nt; t += 256) {
                local_max = sycl::maximum<float>()(local_max, scores[t]);
            }
            float mx = sycl::reduce_over_group(item.get_group(), local_max, sycl::maximum<float>());
            
            float local_sum = 0;
            for(int t = tid; t < nt; t += 256) {
                float e = sycl::exp(scores[t] - mx);
                scores[t] = e;
                local_sum += e;
            }
            float sum = sycl::reduce_over_group(item.get_group(), local_sum, sycl::plus<float>());
            float inv = 1.f / sum;
            
            for(int t = tid; t < nt; t += 256) {
                scores[t] *= inv;
            }
            item.barrier(sycl::access::fence_space::local_space);
            
            for(int k = tid; k < K; k += 256) {
                float a = 0;
                for(int t = 0; t < nt; t++) {
                    a += scores[t] * latent[(size_t)t * K + k];
                }
                cl[k] = a;
            }
            item.barrier(sycl::access::fence_space::local_space);
            
            for(int v = tid; v < V; v += 256) {
                int row = rbase + Q + v;
                float a = 0;
                for(int k = 0; k < K; k++) {
                    a += cl[k] * weight_at(weights, fmt, (size_t)row * rb, k);
                }
                ctx[((size_t)s * H + h) * V + v] = a * (fmt ? wscale[row] : 1.f);
            }
        });
    });
}
extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev, const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) {
    if(!w||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc = find_ctx(w->device);
    if (!dc) return 0;
    attention_absorb_batch_kernel(*dc->q, ctx_dev, q_dev, latent_dev, rope_dev, w->weights, w->scales, w->fmt, S, H, Q, R, V, K, T, scale);
    dc->q->wait();
    return 1;
}

static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj, float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q, int R,int V,int K,int T,float scale) {
    if(!w||!out||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||T<1||w->I!=K||w->O!=H*(Q+V)) return 0;
    DeviceContext *dc = find_ctx(w->device);
    if (!dc) return 0;

    size_t qb = (size_t)S * H * (Q + R) * sizeof(float);
    size_t lb = (size_t)T * K * sizeof(float);
    size_t rb_rope = (size_t)T * R * sizeof(float);
    size_t cb = (size_t)S * H * V * sizeof(float);

    if (!reserve_buf(dc, &dc->aq, &dc->aq_cap, qb) || !reserve_buf(dc, &dc->al, &dc->al_cap, lb) ||
        !reserve_buf(dc, &dc->ar, &dc->ar_cap, rb_rope) || !reserve_buf(dc, &dc->ac, &dc->ac_cap, cb)) return 0;

    dc->q->memcpy(dc->aq, q, qb);
    dc->q->memcpy(dc->al, latent, lb);
    dc->q->memcpy(dc->ar, rope, rb_rope);
    dc->q->wait();

    attention_absorb_batch_kernel(*dc->q, dc->ac, dc->aq, dc->al, dc->ar, w->weights, w->scales, w->fmt, S, H, Q, R, V, K, T, scale);

    float *src = dc->ac;
    size_t ob = cb;
    if (proj) {
        ob = (size_t)S * proj->O * sizeof(float);
        if (!reserve_buf(dc, &dc->y, &dc->y_cap, ob)) return 0;
        coli_sycl_pipe_gemm(proj, dc->y, dc->ac, S);
        src = dc->y;
    }

    dc->q->memcpy(out, src, ob).wait();
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int S, int H,int Q,int R,int V,int K,int T, float attention_scale) {
    return attention_absorb_batch_run(kv_b, nullptr, ctx, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}
extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q,const float *latent, const float *rope,int S,int H,int Q,int R, int V,int K,int T,float attention_scale) {
    return attention_absorb_batch_run(kv_b, o_proj, out, q, latent, rope, S, H, Q, R, V, K, T, attention_scale);
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent,const float *rope,int H,int Q, int R,int V,int K,int T,float attention_scale) {
    return attention_absorb_batch_run(kv_b, nullptr, ctx, q, latent, rope, 1, H, Q, R, V, K, T, attention_scale);
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q, const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T, float scale) {
    if(!kv_b||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       kv_b->I!=K||kv_b->O!=H*(Q+V))return 0;
    DeviceContext *dc = find_ctx(kv_b->device);
    if (!dc) return 0;
    
    size_t qb = (size_t)H * (Q + R) * sizeof(float);
    size_t cb = (size_t)H * V * sizeof(float);
    if (!reserve_buf(dc, &dc->aq, &dc->aq_cap, qb) || !reserve_buf(dc, &dc->ac, &dc->ac_cap, cb)) return 0;
    
    dc->q->memcpy(dc->aq, q, qb).wait();
    attention_absorb_batch_kernel(*dc->q, dc->ac, dc->aq, latent_dev, rope_dev, kv_b->weights, kv_b->scales, kv_b->fmt, 1, H, Q, R, V, K, T, scale);
    dc->q->memcpy(ctx, dc->ac, cb).wait();
    return 1;
}

extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) {
    if(!kv_b||!o_proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||kv_b->I!=K||kv_b->O!=H*(Q+V)||
       o_proj->device!=kv_b->device||o_proj->I!=H*V)return 0;
    DeviceContext *dc = find_ctx(kv_b->device);
    if (!dc) return 0;
    
    size_t cb = (size_t)S * H * V * sizeof(float);
    if (!reserve_buf(dc, &dc->ac, &dc->ac_cap, cb)) return 0;
    
    attention_absorb_batch_kernel(*dc->q, dc->ac, q_dev, latent_dev, rope_dev, kv_b->weights, kv_b->scales, kv_b->fmt, S, H, Q, R, V, K, T, scale);
    
    size_t ob = (size_t)S * o_proj->O * sizeof(float);
    if (!reserve_buf(dc, &dc->y, &dc->y_cap, ob)) return 0;
    
    coli_sycl_pipe_gemm(o_proj, dc->y, dc->ac, S);
    dc->q->memcpy(out, dc->y, ob).wait();
    return 1;
}
extern "C" COLI_SYCL_DLLEXPORT int coli_sycl_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj, float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev, int S,int H,int Q,int R,int V,int K,int T,float scale) {
    if(!kv_b||!o_proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||kv_b->I!=K||kv_b->O!=H*(Q+V)||
       o_proj->device!=kv_b->device||o_proj->I!=H*V)return 0;
    DeviceContext *dc = find_ctx(kv_b->device);
    if (!dc) return 0;
    
    size_t cb = (size_t)S * H * V * sizeof(float);
    if (!reserve_buf(dc, &dc->ac, &dc->ac_cap, cb)) return 0;
    
    attention_absorb_batch_kernel(*dc->q, dc->ac, q_dev, latent_dev, rope_dev, kv_b->weights, kv_b->scales, kv_b->fmt, S, H, Q, R, V, K, T, scale);
    coli_sycl_pipe_gemm(o_proj, out_dev, dc->ac, S);
    dc->q->wait();
    return 1;
}
