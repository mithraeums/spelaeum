/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* gpu_web.c — the engine's gpu.h for the page: buffers and kernels on WebGPU,
 * through web/gpu.js imported as module "gpu". Reading a buffer back is the one
 * call that waits (gpu.js makes it a JSPI suspending import); everything else
 * only queues. With no WebGPU, or no JSPI to wait with, gpu.js names no device
 * and the trainer stays on the CPU. Only the training half of gpu.h is here: the
 * page runs no model, so the matvecs and the per-product offload say "the CPU". */
#include "gpu.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __wasm__
#define IMP(n) __attribute__((import_module("gpu"), import_name(n)))
#else
#define IMP(n)   /* werror-check compiles web/ natively too */
#endif
IMP("name")   int  js_name(char *buf, int cap);              /* its length, 0 for no device */
IMP("new")    int  js_new(uint32_t bytes);                   /* a handle, 0 when it cannot be made */
IMP("del")    void js_del(int h);
IMP("put")    int  js_put(int h, uint32_t off, const void *src, uint32_t n);
IMP("get")    int  js_get(int h, uint32_t off, void *dst, uint32_t n);
IMP("launch") int  js_launch(int k, uint32_t gx, uint32_t gy, uint32_t gz, int n, const uint32_t *args);

struct gpu_mem { int h; size_t bytes; };

static int state;            /* 0 not asked, 1 a device, -1 none */
static char name[128];
static long gemms;

const char *gpu_name(void) {
	if (!state) {
		const char *e = getenv("HAKM_GPU");
		int n = e && !strcmp(e, "0") ? 0 : js_name(name, (int)sizeof name - 1);
		if (n > 0) name[n] = 0;
		state = n > 0 ? 1 : -1;
	}
	return state > 0 ? name : NULL;
}

gpu_mem *gpu_new(size_t bytes) {
	if (!gpu_name() || !bytes || bytes > UINT32_MAX) return NULL;
	gpu_mem *b = calloc(1, sizeof *b);
	if (!b) return NULL;
	if (!(b->h = js_new((uint32_t)bytes))) { free(b); return NULL; }
	b->bytes = bytes;
	return b;
}

void gpu_del(gpu_mem *b) {
	if (!b) return;
	js_del(b->h);
	free(b);
}

static int fits(const gpu_mem *b, size_t off, size_t n) { return b && state > 0 && off <= b->bytes && n <= b->bytes - off; }
int gpu_put(gpu_mem *b, size_t off, const void *src, size_t n) {
	if (!fits(b, off, n)) return -1;
	return n ? js_put(b->h, (uint32_t)off, src, (uint32_t)n) : 0;
}
int gpu_get(gpu_mem *b, size_t off, void *dst, size_t n) {
	if (!fits(b, off, n)) return -1;
	return n ? js_get(b->h, (uint32_t)off, dst, (uint32_t)n) : 0;
}
int gpu_zero(gpu_mem *b, size_t off, size_t n) {
	if (!fits(b, off, n) || off % 4 || n % 4) return -1;
	if (!n) return 0;
	gpu_arg a[] = { GP(b, off / 4), GL((int64_t)(n / 4)) };
	return gpu_launch(GK_ZERO, n / 4, 1, 1, 2, a);
}

/* every argument as 4 words for gpu.js: type, then a buffer's handle and offset, or the value */
int gpu_launch(int k, size_t gx, size_t gy, size_t gz, int n, const gpu_arg *a) {
	if (state <= 0 || k < 0 || k >= GK_N || !gx || !gy || !gz || n < 0 || n > 32 ||
	    gx > UINT32_MAX || gy > UINT32_MAX || gz > UINT32_MAX) return -1;
	uint32_t w[32 * 4];
	for (int i = 0; i < n; i++) {
		uint32_t *o = w + 4 * i;
		o[0] = (uint32_t)a[i].t; o[1] = o[2] = o[3] = 0;
		if (a[i].t == GA_PTR) {
			if (!a[i].m || a[i].off < 0 || (size_t)a[i].off > a[i].m->bytes / 4) return -1;
			o[1] = (uint32_t)a[i].m->h; o[2] = (uint32_t)a[i].off;
		} else if (a[i].t == GA_INT) o[1] = (uint32_t)a[i].i;
		else if (a[i].t == GA_FLOAT) memcpy(&o[1], &a[i].f, 4);
		else { if (a[i].off < 0 || (unsigned long long)a[i].off > UINT32_MAX) return -1; o[1] = (uint32_t)a[i].off; }   /* long is 32 bits here */
	}
	int rc = js_launch(k, (uint32_t)gx, (uint32_t)gy, (uint32_t)gz, n, w);
	if (!rc && k == GK_GEMM) gemms++;
	return rc;
}

/* the inference half: the page does not run a model */
int gpu_matmul(float *out, const float *x, const hakm_model *m, const mlf2_tensor *w,
               const float *bias, int n_out, int n_in, int B) {
	(void)out; (void)x; (void)m; (void)w; (void)bias; (void)n_out; (void)n_in; (void)B;
	return -1;
}
int gpu_gemm(int M, int N, int K, const float *A, size_t sm, size_t sk, const float *B, size_t ldb,
             float *C, size_t ldc, int add) {
	(void)M; (void)N; (void)K; (void)A; (void)sm; (void)sk; (void)B; (void)ldb; (void)C; (void)ldc; (void)add;
	return -1;
}
void gpu_forget(const void *base, size_t n) { (void)base; (void)n; }
size_t gpu_resident(void) { return 0; }
long gpu_gemm_count(void) { return gemms; }
int gpu_broken(void) { return 0; }
