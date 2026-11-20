/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* spel.c (web) — the chain's own C for a page, built for wasm32-wasip1. The page keeps
 * the chain directory in an in-memory file system (web/wasi.js), does the HTTP itself
 * (web/trainer.js, the same steps as net_sync), and runs one of these per step:
 *
 *   key     SEEDHEX               the public key, in hex
 *   start   DIR                   a new copy's tip at genesis (data and genesis checked)
 *   advance DIR                   apply the whole blocks above the copy's tip
 *   heads   DIR FROM FILE [UPTO]  head parts from a /heads answer into DIR/blocks, none past UPTO
 *   join    DIR MODEL             take DIR/MODEL as tip.mlf2 when its hash is the last head's
 *   prune   DIR KEEP              cut the floats off all but the last KEEP blocks
 *   train   DIR NAME SEEDHEX      one job; prints the delta's hex
 *   check   DIR ROUND DELTA SEEDHEX  a closed round's delta checked on DIR's tip; prints the
 *                                 signed verdict (136 bytes) in hex, to POST to /verdict
 *   credit  DIR KEYHEX            a public key's credit as one JSON line: passed,
 *                                 merged, refused, ops, grade (null for none), next
 *   window  NTOK SHARDS STRIPE S R T  where a job's window starts (ch_window), held by
 *                                 test.mjs to the terminal's 64-bit positions past 2^31 tokens
 *   gpu                           the GPU a job trains on (web/gpu.js), or "cpu"
 *   gpucheck                      training on the GPU held to the CPU (test-gpu.mjs): exit 0
 *                                 when it agrees, 1 when not, 2 with no GPU
 */
#include "../src/chain.h"
#include "../src/sha256.h"
#include "train.h"
#include "gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int seed_of(const char *s, ch_key *k) {
	unsigned char seed[32];
	if (strlen(s) != 64) return -1;
	for (int i = 0; i < 32; i++) {
		unsigned v;
		if (sscanf(s + 2 * i, "%2x", &v) != 1) return -1;
		seed[i] = (unsigned char)v;
	}
	ch_key_from(seed, k);
	return 0;
}

/* a training run's loss and every grad with the products on the GPU against the CPU, then
   three whole steps (the last at a smaller batch) agreeing on the loss, and a forward pass
   alone on the CPU bit for bit (the chain's losses): test_gpu.c's train_one, for the page */
static int fails;
static void ck(int ok, const char *what) { printf("  %s %s\n", ok ? "ok  " : "FAIL", what); fails += !ok; }
static void train_one(const tr_config *c, int Bn, const char *tn) {
	tr_model *a = tr_new(c, 5), *b = tr_new(c, 5);
	int T = c->ctx, n = Bn * T, *in = malloc(n * sizeof(int)), *tg = malloc(n * sizeof(int));
	if (!a || !b || !in || !tg) { ck(0, "memory"); return; }
	uint64_t r = 7;
	for (int i = 0; i < n; i++) { r = r * 6364136223846793005ULL + 1442695040888963407ULL; in[i] = (int)((r >> 33) % c->vocab); tg[i] = (int)((r >> 13) % c->vocab); }
	char what[160];
	tr_use_gpu(0);
	double la = tr_run(a, in, tg, Bn, T, 1.0f);
	tr_use_gpu(1);
	long g0 = gpu_gemm_count();
	double lb = tr_run(b, in, tg, Bn, T, 1.0f);
	snprintf(what, sizeof what, "%s: loss %.9g on the GPU (%ld products), %.9g on the CPU", tn, lb, gpu_gemm_count() - g0, la);
	ck(gpu_gemm_count() > g0 && la > 0 && fabs(la - lb) <= 1e-5 * la, what);
	const float *ga = tr_grads(a), *gb = tr_grads(b);
	double gmax = 0, dmax = 0;
	for (size_t i = 0; i < tr_n_params(a); i++) {
		if (fabs(ga[i]) > gmax) gmax = fabs(ga[i]);
		if (fabs(ga[i] - gb[i]) > dmax) dmax = fabs(ga[i] - gb[i]);
	}
	snprintf(what, sizeof what, "%s: grads differ by at most %.3g of their largest %.3g", tn, dmax, gmax);
	ck(dmax <= 1e-4 * gmax, what);
	for (int st = 0; st < 3; st++) {
		int bn = st == 2 ? Bn - 1 : Bn;
		for (int k = 0; k < 2; k++) {
			tr_model *m = k ? b : a;
			tr_use_gpu(k);
			tr_zero_grad(m);
			tr_run(m, in, tg, bn, T, 1.0f / (bn * T));
			tr_clip(m, 0.5);
			tr_adamw(m, 1e-3f, 0.1f, 0.9f, 0.95f, 1e-8f);
		}
	}
	g0 = gpu_gemm_count();
	double fa = tr_run(a, in, tg, Bn, T, 0), fb = tr_run(b, in, tg, Bn, T, 0), f2 = tr_run(a, in, tg, Bn, T, 0);
	snprintf(what, sizeof what, "%s: after 3 steps the loss is %.6g on the GPU's weights, %.6g on the CPU's", tn, fb, fa);
	ck(fabs(fa - fb) <= 1e-3 * fa && fa < la * 0.999, what);
	snprintf(what, sizeof what, "%s: a forward pass alone stays on the CPU, bit for bit", tn);
	ck(fa == f2 && gpu_gemm_count() == g0, what);
	tr_free(a); tr_free(b); free(in); free(tg);
}

int main(int argc, char **argv) {
	const char *cmd = argc > 1 ? argv[1] : "";
	char hx[65];
	ch_key k;
	if (!strcmp(cmd, "key") && argc == 3) {
		if (seed_of(argv[2], &k)) return 2;
		sha256_hex(k.pk, hx);
		printf("%s\n", hx);
		return 0;
	}
	if (!strcmp(cmd, "window") && argc == 8) {
		int64_t w = ch_window(strtoll(argv[2], NULL, 10), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]),
		                      strtoull(argv[6], NULL, 10), atoi(argv[7]));
		printf("%lld\n", (long long)w);
		return 0;
	}
	if (!strcmp(cmd, "start") && argc == 3) return ch_start(argv[2]) ? 1 : 0;
	if (!strcmp(cmd, "advance") && argc == 3) return ch_advance(argv[2]) ? 1 : 0;
	if (!strcmp(cmd, "join") && argc == 4) return ch_join(argv[2], argv[3]) ? 1 : 0;
	if (!strcmp(cmd, "prune") && argc == 4) return ch_prune(argv[2], strtoull(argv[3], NULL, 10), 1) ? 1 : 0;
	if (!strcmp(cmd, "heads") && (argc == 5 || argc == 6)) {
		FILE *f = fopen(argv[4], "rb");
		if (!f) return 1;
		size_t cap = 1 << 16, len = 0, r;
		unsigned char *b = malloc(cap);
		while (b && (r = fread(b + len, 1, cap - len, f)) > 0) {
			len += r;
			if (len == cap) { unsigned char *nb = realloc(b, cap *= 2); if (!nb) { free(b); b = NULL; } else b = nb; }
		}
		fclose(f);
		uint64_t got = 0;
		int rc = !b || ch_put_heads(argv[2], b, len, strtoull(argv[3], NULL, 10), argc == 6 ? strtoull(argv[5], NULL, 10) : UINT64_MAX, &got) || !got;
		free(b);
		if (!rc) printf("%llu\n", (unsigned long long)got);
		return rc;
	}
	if (!strcmp(cmd, "gpu")) { printf("%s\n", tr_gpu_name() ? tr_gpu_name() : "cpu"); return 0; }
	if (!strcmp(cmd, "gpucheck")) {
		if (!tr_gpu_name()) { printf("gpucheck: no GPU\n"); return 2; }
		printf("gpucheck on %s\n", tr_gpu_name());
		/* the second folds its widest launches past 65535 work-groups a side, and groups its kv heads */
		tr_config dense = { 300, 64, 2, 4, 2, 16, 128, 32, 10000.0f, 1e-5f, 0, 0, 0 };
		tr_config wide = { 300, 64, 1, 2, 1, 32, 8256, 128, 10000.0f, 1e-5f, 0, 0, 0 };
		train_one(&dense, 3, "dense");
		tr_config mid = { 300, 128, 1, 4, 2, 32, 264, 64, 10000.0f, 1e-5f, 0, 0, 0 };   /* products past 128 each way, not whole tiles */
		train_one(&wide, 4, "wide");
		train_one(&mid, 5, "mid");
		printf(fails ? "gpucheck: %d FAILED\n" : "gpucheck: all agree\n", fails);
		return fails ? 1 : 0;
	}
	if (!strcmp(cmd, "train") && argc == 5) {
		if (seed_of(argv[4], &k) || ch_train(argv[2], argv[3], &k, hx)) return 1;
		printf("%s\n", hx);
		return 0;
	}
	if (!strcmp(cmd, "check") && argc == 6) {
		unsigned char parent[32];
		ch_verdict v;
		if (seed_of(argv[5], &k)) return 2;
		if (ch_check(argv[2], argv[3], argv[4], parent, &v)) return 1;
		ch_verdict_sign(&k, parent, &v);
		const unsigned char *b = (const unsigned char *)&v;
		for (size_t i = 0; i < sizeof v; i++) printf("%02x", b[i]);
		printf("\n");
		return 0;
	}
	if (!strcmp(cmd, "credit") && argc == 4) {
		unsigned char want[32];
		for (int i = 0; i < 32; i++) {
			unsigned v;
			if (strlen(argv[3]) != 64 || sscanf(argv[3] + 2 * i, "%2x", &v) != 1) return 2;
			want[i] = (unsigned char)v;
		}
		ch_credit *cr = NULL, z = { .passed = 0 }, *e = &z;
		double next;
		long n = ch_credits(argv[2], &cr, NULL);
		if (n < 0) return 1;
		for (long i = 0; i < n; i++) if (!memcmp(cr[i].key, want, 32)) e = &cr[i];
		const char *g;
		ch_grade(e->ops, &g, &next);
		printf("{\"passed\":%llu,\"merged\":%llu,\"refused\":%llu,\"ops\":%.17g,%s%s%s,\"next\":%.17g}\n",
		       (unsigned long long)e->passed, (unsigned long long)e->merged, (unsigned long long)e->refused, e->ops,
		       g ? "\"grade\":\"" : "\"grade\":null", g ? g : "", g ? "\"" : "", next);
		free(cr);
		return 0;
	}
	fprintf(stderr, "usage: key SEEDHEX | start DIR | advance DIR | heads DIR FROM FILE [UPTO] | join DIR MODEL | prune DIR KEEP | train DIR NAME SEEDHEX | check DIR ROUND DELTA SEEDHEX | credit DIR KEYHEX\n");
	return 2;
}
