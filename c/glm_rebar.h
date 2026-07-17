#ifndef GLM_REBAR_H
#define GLM_REBAR_H
#include <stddef.h>
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif
#ifdef COLI_SYCL
#include "backend_sycl.h"
#endif
#ifdef COLI_METAL
#include "backend_metal.h"
#endif
#ifdef COLI_VULKAN
#include "backend_vulkan.h"
#endif


static inline void* engine_alloc_mapped(size_t bytes, void** device_ptr) {
#ifdef COLI_CUDA
    if (g_cuda_enabled) return coli_cuda_alloc_mapped(bytes, device_ptr);
#endif
#ifdef COLI_SYCL
    return coli_sycl_alloc_mapped(bytes, device_ptr);
#endif
#ifdef COLI_METAL
    if (g_metal_enabled) return coli_metal_alloc_mapped(bytes, device_ptr);
#endif
#ifdef COLI_VULKAN
    return coli_vulkan_alloc_mapped(bytes, device_ptr);
#endif
    return NULL;
}

static inline void engine_free_mapped(void* host_ptr) {
#ifdef COLI_CUDA
    if (g_cuda_enabled) { coli_cuda_free_mapped(host_ptr); return; }
#endif
#ifdef COLI_SYCL
    { coli_sycl_free_mapped(host_ptr); return; }
#endif
#ifdef COLI_METAL
    if (g_metal_enabled) { coli_metal_free_mapped(host_ptr); return; }
#endif
#ifdef COLI_VULKAN
    { coli_vulkan_free_mapped(host_ptr); return; }
#endif
}
#endif
