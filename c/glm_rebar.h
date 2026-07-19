#ifndef GLM_REBAR_H
#define GLM_REBAR_H
#include <stddef.h>

void* engine_alloc_mapped(size_t bytes, void** device_ptr);
void engine_free_mapped(void* host_ptr);

#endif
