#ifndef FFN_ASYNC_BUFFER_H
#define FFN_ASYNC_BUFFER_H

#if defined(__linux__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <liburing.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#ifndef O_DIRECT
#define O_DIRECT 040000
#endif

typedef struct {
    int fd;
    void* buf[2];
    int active_idx;
    int prefetch_idx;
    struct io_uring ring;
} ffn_async_buffer;

static inline int ffn_async_init(ffn_async_buffer* ab, const char* filepath, size_t buf_size) {
    ab->active_idx = 0;
    ab->prefetch_idx = 1;
    ab->fd = open(filepath, O_RDONLY | O_DIRECT);
    if (ab->fd < 0) {
        ab->fd = open(filepath, O_RDONLY);
        if (ab->fd < 0) return -1;
    }
    if (io_uring_queue_init(1, &ab->ring, 0) < 0) { close(ab->fd); return -1; }
    if (posix_memalign(&ab->buf[0], 4096, buf_size) != 0) { io_uring_queue_exit(&ab->ring); close(ab->fd); return -1; }
    if (posix_memalign(&ab->buf[1], 4096, buf_size) != 0) { free(ab->buf[0]); io_uring_queue_exit(&ab->ring); close(ab->fd); return -1; }
    return 0;
}

static inline void ffn_async_prefetch(ffn_async_buffer* ab, uint64_t offset, size_t size) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ab->ring);
    io_uring_prep_read(sqe, ab->fd, ab->buf[ab->prefetch_idx], size, offset);
    io_uring_submit(&ab->ring);
}

static inline void ffn_async_swap(ffn_async_buffer* ab) {
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ab->ring, &cqe);
    io_uring_cqe_seen(&ab->ring, cqe);
    int tmp = ab->active_idx;
    ab->active_idx = ab->prefetch_idx;
    ab->prefetch_idx = tmp;
}

static inline void ffn_async_destroy(ffn_async_buffer* ab) {
    if (ab->fd >= 0) close(ab->fd);
    io_uring_queue_exit(&ab->ring);
    if (ab->buf[0]) free(ab->buf[0]);
    if (ab->buf[1]) free(ab->buf[1]);
}

#elif defined(_WIN32)
#include <windows.h>
#include <io.h>
typedef struct {
    int fd;
    void* buf[2];
    int active_idx;
    int prefetch_idx;
    HANDLE iocp;
    OVERLAPPED ov;
} ffn_async_buffer;

static inline int ffn_async_init(ffn_async_buffer* ab, const char* filepath, size_t buf_size) {
    ab->active_idx = 0;
    ab->prefetch_idx = 1;
    HANDLE hFile = CreateFileA(filepath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return -1;
    ab->fd = _open_osfhandle((intptr_t)hFile, _O_RDONLY);
    ab->iocp = CreateIoCompletionPort(hFile, NULL, 0, 0);
    ab->buf[0] = _aligned_malloc(buf_size, 4096);
    ab->buf[1] = _aligned_malloc(buf_size, 4096);
    memset(&ab->ov, 0, sizeof(OVERLAPPED));
    return 0;
}

static inline void ffn_async_prefetch(ffn_async_buffer* ab, uint64_t offset, size_t size) {
    ab->ov.Offset = offset & 0xFFFFFFFF;
    ab->ov.OffsetHigh = offset >> 32;
    HANDLE hFile = (HANDLE)_get_osfhandle(ab->fd);
    ReadFile(hFile, ab->buf[ab->prefetch_idx], (DWORD)size, NULL, &ab->ov);
}

static inline void ffn_async_swap(ffn_async_buffer* ab) {
    DWORD bytes;
    HANDLE hFile = (HANDLE)_get_osfhandle(ab->fd);
    GetOverlappedResult(hFile, &ab->ov, &bytes, TRUE);
    int tmp = ab->active_idx;
    ab->active_idx = ab->prefetch_idx;
    ab->prefetch_idx = tmp;
}

static inline void ffn_async_destroy(ffn_async_buffer* ab) {
    if (ab->fd >= 0) _close(ab->fd);
    if (ab->iocp) CloseHandle(ab->iocp);
    if (ab->buf[0]) _aligned_free(ab->buf[0]);
    if (ab->buf[1]) _aligned_free(ab->buf[1]);
}

#else
/* Fallback for MacOS/etc */
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
typedef struct {
    int fd;
    void* buf[2];
    int active_idx;
    int prefetch_idx;
} ffn_async_buffer;

static inline int ffn_async_init(ffn_async_buffer* ab, const char* filepath, size_t buf_size) {
    ab->active_idx = 0;
    ab->prefetch_idx = 1;
    ab->fd = open(filepath, O_RDONLY);
    if (ab->fd < 0) return -1;
    if (posix_memalign(&ab->buf[0], 4096, buf_size) != 0) { close(ab->fd); return -1; }
    if (posix_memalign(&ab->buf[1], 4096, buf_size) != 0) { free(ab->buf[0]); close(ab->fd); return -1; }
    return 0;
}

static inline void ffn_async_prefetch(ffn_async_buffer* ab, uint64_t offset, size_t size) {
    pread(ab->fd, ab->buf[ab->prefetch_idx], size, offset);
}

static inline void ffn_async_swap(ffn_async_buffer* ab) {
    int tmp = ab->active_idx;
    ab->active_idx = ab->prefetch_idx;
    ab->prefetch_idx = tmp;
}

static inline void ffn_async_destroy(ffn_async_buffer* ab) {
    if (ab->fd >= 0) close(ab->fd);
    if (ab->buf[0]) free(ab->buf[0]);
    if (ab->buf[1]) free(ab->buf[1]);
}
#endif
#endif
