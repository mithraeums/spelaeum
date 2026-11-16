/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* chain.c — Spelaeum: block mining applied to training. See chain.h for the shape. */
#include "chain.h"
#include "sha256.h"
#include "train.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(sizeof(ch_dhead) == 96, "ch_dhead is written as is");
_Static_assert(sizeof(ch_bhead) == 120, "ch_bhead is written as is");
_Static_assert(sizeof(ch_contrib) == 80, "ch_contrib is written as is");

#define PATHN 4096
static int pj(char *o, const char *dir, const char *name) {
	int k = snprintf(o, PATHN, "%s/%s", dir, name);
	if (k < 0 || k >= PATHN) { fprintf(stderr, "chain: path too long: %s/%s\n", dir, name); return -1; }
	return 0;
}

static uint64_t mix(uint64_t z) {
	z += 0x9E3779B97F4A7C15ULL;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}
static uint64_t h64(const unsigned char *h) {
	uint64_t v = 0;
	for (int i = 0; i < 8; i++) v |= (uint64_t)h[i] << (8 * i);
	return v;
}
/* 64 hex chars; what follows them is the caller's to check */
static int unhex(const char *s, unsigned char out[32]) {
	for (int i = 0; i < 32; i++) {
		int v = 0;
		for (int j = 0; j < 2; j++) {
			char c = s[2 * i + j];
			int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
			if (d < 0) return -1;
			v = v * 16 + d;
		}
		out[i] = (unsigned char)v;
	}
	return 0;
}
static int miner_ok(const char *s) {
	size_t n = strnlen(s, CH_MINER);
	if (n == 0 || n >= CH_MINER) return 0;
	for (size_t i = 0; i < n; i++) {
		char c = s[i];
		if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) return 0;
	}
	return 1;
}

/* whole file into memory, and its sha256 */
static unsigned char *slurp_hash(const char *path, size_t *len, unsigned char hash[32]) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	struct stat st;
	unsigned char *b = NULL;
	if (fstat(fileno(f), &st) == 0 && st.st_size >= 0 && (b = malloc((size_t)st.st_size + 1)) &&
	    fread(b, 1, (size_t)st.st_size, f) == (size_t)st.st_size) {
		*len = (size_t)st.st_size;
		if (hash) sha256(b, *len, hash);
	} else { free(b); b = NULL; }
	fclose(f);
	return b;
}

/* write head + parts through <path>.part and a rename, hashing what goes out */
typedef struct { const void *p; size_t n; } part;
static int write_hashed(const char *path, const part *ps, int np, unsigned char hash[32]) {
	char tmp[PATHN + 8];
	snprintf(tmp, sizeof tmp, "%s.part", path);
	FILE *f = fopen(tmp, "wb");
	if (!f) { perror(tmp); return -1; }
	sha256_ctx c; sha256_init(&c);
	int ok = 1;
	for (int i = 0; i < np && ok; i++) {
		if (!ps[i].n) continue;
		if (fwrite(ps[i].p, 1, ps[i].n, f) != ps[i].n) ok = 0;
		sha256_add(&c, ps[i].p, ps[i].n);
	}
	if (fclose(f) != 0) ok = 0;
	if (!ok || rename(tmp, path) != 0) { fprintf(stderr, "chain: could not write %s\n", path); remove(tmp); return -1; }
	if (hash) sha256_done(&c, hash);
	return 0;
}

/* ── config ─────────────────────────────────────────────────────────────── */

/* experts only when > 0, so a dense chain hashes as it always has */
static void params_text(const ch_params *p, char *out, size_t cap) {
	int k = snprintf(out, cap, "shards %d\nstripe %d\nsteps %d\nbatch %d\nlr %.9g\neval %d\ndifficulty %.17g\n",
	                 p->shards, p->stripe, p->steps, p->batch, (double)p->lr, p->eval_windows, p->difficulty);
	if (p->experts > 0 && k > 0 && (size_t)k < cap) k += snprintf(out + k, cap - (size_t)k, "experts %d\n", p->experts);
	if (p->trunk > 0 && k > 0 && (size_t)k < cap) snprintf(out + k, cap - (size_t)k, "trunk %d\n", p->trunk);
}

/* genesis = sha256(params text, data hash, sha256 of genesis.mlf2) */
static int genesis_hash(const char *dir, const ch_params *p, const unsigned char data[32], unsigned char out[32]) {
	char path[PATHN], txt[512];
	if (pj(path, dir, "genesis.mlf2")) return -1;
	size_t len; unsigned char fh[32];
	unsigned char *b = slurp_hash(path, &len, fh);
	if (!b) { fprintf(stderr, "chain: cannot read %s\n", path); return -1; }
	free(b);
	params_text(p, txt, sizeof txt);
	sha256_ctx c; sha256_init(&c);
	sha256_add(&c, "spelaeum genesis\n", 17);
	sha256_add(&c, txt, strlen(txt)); sha256_add(&c, data, 32); sha256_add(&c, fh, 32);
	sha256_done(&c, out);
	return 0;
}

int ch_load_config(const char *dir, ch_config *c) {
	char path[PATHN], line[256], key[32], val[128];
	if (pj(path, dir, "chain.cfg")) return -1;
	FILE *f = fopen(path, "r");
	if (!f) { fprintf(stderr, "chain: %s is not a chain (no chain.cfg)\n", dir); return -1; }
	memset(c, 0, sizeof *c);
	int ver = -1, got = 0;
	while (fgets(line, sizeof line, f)) {
		if (sscanf(line, "%31s %127s", key, val) != 2) continue;
		if      (!strcmp(key, "spelaeum")) ver = atoi(val);
		else if (!strcmp(key, "shards")) { c->p.shards = atoi(val); got |= 1; }
		else if (!strcmp(key, "steps")) { c->p.steps = atoi(val); got |= 2; }
		else if (!strcmp(key, "batch")) { c->p.batch = atoi(val); got |= 4; }
		else if (!strcmp(key, "lr")) { c->p.lr = strtof(val, NULL); got |= 8; }
		else if (!strcmp(key, "eval")) { c->p.eval_windows = atoi(val); got |= 16; }
		else if (!strcmp(key, "difficulty")) { c->p.difficulty = strtod(val, NULL); got |= 32; }
		else if (!strcmp(key, "data") && strlen(val) == 64 && !unhex(val, c->data)) got |= 64;
		else if (!strcmp(key, "genesis") && strlen(val) == 64 && !unhex(val, c->genesis)) got |= 128;
		else if (!strcmp(key, "stripe")) { c->p.stripe = atoi(val); got |= 256; }
		else if (!strcmp(key, "experts")) c->p.experts = atoi(val);
		else if (!strcmp(key, "trunk")) c->p.trunk = atoi(val);
	}
	fclose(f);
	if (ver != CH_VERSION || got != 511 || c->p.shards < 2 || c->p.stripe < 4 || c->p.steps < 1 || c->p.batch < 1 || c->p.eval_windows < 1 ||
	    c->p.experts < 0 || c->p.trunk < 0 || (c->p.trunk && !c->p.experts)) {
		fprintf(stderr, "chain: %s is not a chain this build reads (version %d)\n", path, ver);
		return -1;
	}
	return 0;
}

/* ── training tokens ────────────────────────────────────────────────────── */

typedef struct { char magic[4]; uint32_t version; uint64_t n; } ids_head;
typedef struct { const int *ids; long n; void *map; size_t size; } ch_ids;

static int ids_open(const char *dir, ch_ids *d) {
	char path[PATHN];
	if (pj(path, dir, "ids.bin")) return -1;
	int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st) != 0) { perror(path); if (fd >= 0) close(fd); return -1; }
	d->size = (size_t)st.st_size;
	d->map = d->size >= sizeof(ids_head) ? mmap(NULL, d->size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
	close(fd);
	if (d->map == MAP_FAILED) { fprintf(stderr, "chain: cannot map %s\n", path); return -1; }
	const ids_head *h = d->map;
	if (memcmp(h->magic, "HKID", 4) || h->version != CH_VERSION || h->n > (d->size - sizeof *h) / 4 ||
	    sizeof *h + h->n * 4 != d->size) {
		fprintf(stderr, "chain: %s is damaged\n", path);
		munmap(d->map, d->size); return -1;
	}
	d->ids = (const int *)((const char *)d->map + sizeof *h);
	d->n = (long)h->n;
	return 0;
}
static void ids_close(ch_ids *d) { if (d->map) munmap(d->map, d->size); d->map = NULL; }

/* ── the pieces both sides compute ──────────────────────────────────────── */

uint32_t ch_shard(const unsigned char parent[32], const char *miner, int shards) {
	unsigned char h[32];
	sha256_ctx c; sha256_init(&c);
	sha256_add(&c, parent, 32); sha256_add(&c, miner, strnlen(miner, CH_MINER));
	sha256_done(&c, h);
	return (uint32_t)(h64(h) % (uint64_t)shards);
}

/* Floyd's sampling per layer: per draws from the layer's own hash, each a new
 * expert, kept sorted as they land */
int ch_units(const unsigned char parent[32], const char *miner, int layers, int experts, int per, uint32_t *out) {
	if (layers < 1 || per < 1 || per > experts) return -1;
	for (int l = 0; l < layers; l++) {
		unsigned char h[32], lb[4] = { (unsigned char)l, (unsigned char)(l >> 8), (unsigned char)(l >> 16), (unsigned char)(l >> 24) };
		sha256_ctx c; sha256_init(&c);
		sha256_add(&c, "spelaeum units\n", 15); sha256_add(&c, parent, 32);
		sha256_add(&c, miner, strnlen(miner, CH_MINER)); sha256_add(&c, lb, 4);
		sha256_done(&c, h);
		uint64_t r = h64(h);
		uint32_t *o = out + (size_t)l * per, u0 = 1 + (uint32_t)l * (uint32_t)experts;
		for (int j = experts - per, n = 0; j < experts; j++, n++) {
			uint32_t t = u0 + (uint32_t)(mix(r + (uint64_t)j) % (uint64_t)(j + 1)), at = (uint32_t)n;
			for (int i = 0; i < n; i++) if (o[i] == t) t = u0 + (uint32_t)j;   /* taken: j, new by construction */
			while (at && o[at - 1] > t) { o[at] = o[at - 1]; at--; }
			o[at] = t;
		}
	}
	return layers * per;
}

/* block h of an expert chain is a trunk round when trunk divides it: every job trains
 * unit 0 (attention, routers, the shared feed-forward) and no expert, so the deltas
 * all collide there and the block is their mean */
static int trunk_round(const ch_params *cp, uint64_t h) { return cp->experts > 0 && cp->trunk > 0 && h % (uint64_t)cp->trunk == 0; }
/* the most units one job has: layers x per on an expert chain, the whole model as one on a dense one */
static int job_size(const tr_model *m, const ch_params *cp) {
	const tr_config *mc = tr_cfg(m);
	if (!cp->experts) return 1;
	if (cp->experts > mc->experts) {
		fprintf(stderr, "chain: %d experts of each layer per job, but the model has %d%s\n", cp->experts, mc->experts, mc->experts ? "" : " (it is dense)");
		return -1;
	}
	return mc->layers * cp->experts;
}
/* the units of a job on block h: how many, and which */
static int job_units(const tr_model *m, const ch_params *cp, uint64_t h, const unsigned char parent[32], const char *miner, uint32_t *un) {
	if (!cp->experts || trunk_round(cp, h)) { un[0] = 0; return 1; }
	return ch_units(parent, miner, tr_cfg(m)->layers, tr_cfg(m)->experts, cp->experts, un);
}
/* where unit u sits: 0 = the trunk [0, expert (0, 0)), 1 + l x E + e = expert (l, e);
 * a dense chain's one unit is the whole model */
static size_t unit_at(const tr_model *m, int moe, uint32_t u, size_t *len) {
	const tr_config *mc = tr_cfg(m);
	if (!moe) { *len = tr_n_params(m); return 0; }
	if (!u) { *len = tr_expert_off(m, 0, 0); return 0; }
	*len = (size_t)3 * mc->ffn * mc->dim;
	return tr_expert_off(m, (int)((u - 1) / (uint32_t)mc->experts), (int)((u - 1) % (uint32_t)mc->experts));
}
/* a packed delta d onto its units in p: p = base + d, p += d when base is NULL, or
 * p = base when d is NULL. Returns the floats it covers. */
static size_t apply(const tr_model *m, int moe, const uint32_t *un, int nu, float *p, const float *base, const float *d) {
	size_t pos = 0, len;
	for (int k = 0; k < nu; k++, pos += len) {
		size_t off = unit_at(m, moe, un[k], &len);
		float *o = p + off;
		if (!d) memcpy(o, base + off, len * sizeof(float));
		else if (base) for (size_t j = 0; j < len; j++) o[j] = base[off + j] + d[pos + j];
		else for (size_t j = 0; j < len; j++) o[j] += d[pos + j];
	}
	return pos;
}
/* every float outside the units, as src has it (zero when src is NULL) */
static size_t units_len(const tr_model *m, int moe, const uint32_t *un, int nu) {
	size_t n = 0, len;
	for (int k = 0; k < nu; k++) { unit_at(m, moe, un[k], &len); n += len; }
	return n;
}
static void outside(const tr_model *m, const uint32_t *un, int nu, float *dst, const float *src) {
	size_t at = 0, len = 0, np = tr_n_params(m);
	for (int k = 0; k <= nu; k++) {
		size_t off = k < nu ? unit_at(m, 1, un[k], &len) : np;
		if (src) memcpy(dst + at, src + at, (off - at) * sizeof(float));
		else memset(dst + at, 0, (off - at) * sizeof(float));
		at = off + len;
	}
}

static int cmp32(const void *a, const void *b) { return memcmp(a, b, 32); }

void ch_seed(const unsigned char parent[32], const unsigned char (*commits)[32], int n, unsigned char out[32]) {
	unsigned char (*s)[32] = n ? malloc((size_t)n * 32) : NULL;
	if (n && s) { memcpy(s, commits, (size_t)n * 32); qsort(s, (size_t)n, 32, cmp32); }
	sha256_ctx c; sha256_init(&c);
	sha256_add(&c, "spelaeum seed\n", 14); sha256_add(&c, parent, 32);
	for (int i = 0; i < n; i++) sha256_add(&c, s ? s[i] : commits[i], 32);
	sha256_done(&c, out);
	free(s);
}

double ch_retarget(double d, int ncontrib, int npassed) {
	if (ncontrib <= 0) return d;
	if (2 * npassed > ncontrib) d *= 1.25;
	else if (2 * npassed < ncontrib) d *= 0.8;
	return d < 1e-6 ? 1e-6 : d;
}

long ch_window(long ntok, int shards, int stripe, int s, uint64_t r, int T) {
	long per = ntok / ((long)stripe * shards);   /* stripes each shard owns */
	long k = (long)(r % (uint64_t)per) * shards + s;
	return k * stripe + (long)(mix(r ^ 0x5851F42D4C957F2DULL) % (uint64_t)(stripe - T - 1));
}

/* a corpus too short for every shard to own a stripe a window fits in */
static int too_short(long ntok, int shards, int stripe, int T) {
	return stripe < T + 2 || ntok / ((long)stripe * shards) < 1;
}

/* the round's windows: n starts of T+1 tokens, from shards nobody mined this round
 * (every shard when the round covered them all) */
static int pick_windows(const unsigned char seed[32], const ch_ids *d, int shards, int stripe,
                        const uint32_t *skip, int nskip, int T, int n, long *at) {
	if (too_short(d->n, shards, stripe, T)) return -1;
	int *ok = malloc((size_t)shards * sizeof *ok), na = 0;
	if (!ok) return -1;
	for (int s = 0; s < shards; s++) {
		int used = 0;
		for (int k = 0; k < nskip; k++) used |= skip[k] == (uint32_t)s;
		if (!used) ok[na++] = s;
	}
	if (!na) for (int s = 0; s < shards; s++) ok[na++] = s;
	uint64_t base = h64(seed);
	for (int i = 0; i < n; i++) {
		uint64_t r = mix(base ^ mix((uint64_t)i));
		at[i] = ch_window(d->n, shards, stripe, ok[r % (uint64_t)na], mix(r), T);
	}
	free(ok);
	return 0;
}

/* nats per token over the windows, forward only; -1 on allocation failure. The
 * callers change the weights through a pointer they already hold, so this drops
 * the trainer's transposed copies first (tr_params does that). */
static double eval_loss(tr_model *m, const int *ids, const long *at, int n, int batch, int T, int *in, int *tg) {
	(void)tr_params(m);
	double nll = 0;
	for (int i = 0; i < n;) {
		int b = 0;
		for (; b < batch && i < n; b++, i++) {
			memcpy(in + (size_t)b * T, ids + at[i], (size_t)T * sizeof(int));
			memcpy(tg + (size_t)b * T, ids + at[i] + 1, (size_t)T * sizeof(int));
		}
		double l = tr_run(m, in, tg, b, T, 0);
		if (l < 0) return -1;
		nll += l;
	}
	return nll / ((double)n * T);
}

/* ── tip and blocks ─────────────────────────────────────────────────────── */

int ch_tip(const char *dir, uint64_t *height, unsigned char hash[32]) {
	char path[PATHN], hex[80];
	unsigned long long h;
	if (pj(path, dir, "tip")) return -1;
	FILE *f = fopen(path, "r");
	int ok = f && fscanf(f, "%llu %70s", &h, hex) == 2 && strlen(hex) == 64 && !unhex(hex, hash);
	if (f) fclose(f);
	if (!ok) { fprintf(stderr, "chain: %s is unreadable\n", path); return -1; }
	*height = h;
	return 0;
}

static int write_tip(const char *dir, uint64_t height, const unsigned char hash[32]) {
	char path[PATHN], hex[65], line[96];
	if (pj(path, dir, "tip")) return -1;
	sha256_hex(hash, hex);
	snprintf(line, sizeof line, "%llu %s\n", (unsigned long long)height, hex);
	part p = { line, strlen(line) };
	return write_hashed(path, &p, 1, NULL);
}

static int block_path(const char *dir, uint64_t h, char *out) {
	char name[48];
	snprintf(name, sizeof name, "blocks/%06llu.blk", (unsigned long long)h);
	return pj(out, dir, name);
}

/* a block file, checked for shape; *buf owns the head, contribs, units (an expert
 * chain's: *nu of them, else none) and delta */
static int read_block(const char *path, int moe, unsigned char **buf, ch_bhead **h, ch_contrib **cs,
                      const uint32_t **un, uint32_t *nu, const float **delta, unsigned char hash[32]) {
	size_t len, at = 0;
	unsigned char *b = slurp_hash(path, &len, hash);
	if (!b) return -1;
	ch_bhead *hd = (ch_bhead *)b;
	uint32_t n = 0;
	int ok = len >= sizeof *hd && !memcmp(hd->magic, "HKBK", 4) && hd->version == (moe ? CH_UNITS : CH_VERSION) &&
	         hd->ncontrib <= 1u << 20 && len - sizeof *hd >= (size_t)hd->ncontrib * sizeof(ch_contrib) + (moe ? 4 : 0);
	if (ok) at = sizeof *hd + (size_t)hd->ncontrib * sizeof(ch_contrib);
	if (ok && moe) { memcpy(&n, b + at, 4); at += 4; ok = n <= (len - at) / 4; }
	if (!ok || hd->n > (len - at - 4 * (size_t)n) / 4 || len != at + 4 * (size_t)n + hd->n * 4) {
		fprintf(stderr, "chain: %s is damaged\n", path);
		free(b); return -1;
	}
	*buf = b; *h = hd;
	*cs = (ch_contrib *)(b + sizeof *hd);
	*un = (const uint32_t *)(b + at); *nu = n;
	*delta = (const float *)(b + at + 4 * (size_t)n);
	return 0;
}

/* the difficulty block h+1 must meet, from block h (or the config at genesis) */
static int next_difficulty(const char *dir, const ch_config *c, uint64_t h, double *d) {
	if (h == 0) { *d = c->p.difficulty; return 0; }
	char path[PATHN];
	unsigned char *b; ch_bhead *hd; ch_contrib *cs; const float *dl; const uint32_t *bu; uint32_t nb; unsigned char hash[32];
	if (block_path(dir, h, path) || read_block(path, c->p.experts > 0, &b, &hd, &cs, &bu, &nb, &dl, hash)) return -1;
	int passed = 0;
	for (uint32_t i = 0; i < hd->ncontrib; i++) passed += cs[i].state != CH_REJECTED;
	*d = ch_retarget(hd->difficulty, (int)hd->ncontrib, passed);
	free(b);
	return 0;
}

static tr_model *load_tip(const char *dir, void **tok, size_t *tlen, int *eos) {
	char path[PATHN];
	if (pj(path, dir, "tip.mlf2")) return NULL;
	return tr_load(path, tok, tlen, eos);
}

/* ── init ───────────────────────────────────────────────────────────────── */

int ch_init(const char *dir, const char *model, const char *opt, const char *data, const ch_params *p) {
	char path[PATHN], txt[512], hx[65], hx2[65];
	if (p->shards < 2 || p->stripe < 4 || p->steps < 1 || p->batch < 1 || p->eval_windows < 1 || !(p->lr > 0) || !(p->difficulty > 0) ||
	    p->experts < 0 || p->trunk < 0 || (p->trunk && !p->experts)) {
		fprintf(stderr, "chain: shards >= 2, stripe >= 4, steps, batch and eval >= 1, lr and difficulty > 0, experts >= 0, trunk >= 0 and only with experts\n");
		return -1;
	}
	/* the model first: one that cannot carry the chain leaves nothing behind */
	void *blob = NULL; size_t blen = 0; int eos = 0;
	tr_model *m = tr_load(model, &blob, &blen, &eos);
	int rc = -1, T = m ? tr_cfg(m)->ctx : 0;
	size_t tlen;
	long n = 0;
	char *text = NULL;
	int *ids = NULL;
	if (!m || job_size(m, p) < 0) goto out;
	if (mkdir(dir, 0755) != 0 && errno != EEXIST) { perror(dir); goto out; }
	if (pj(path, dir, "chain.cfg")) goto out;
	if (access(path, F_OK) == 0) { fprintf(stderr, "chain: %s is a chain already\n", dir); goto out; }
	if (pj(path, dir, "blocks") || (mkdir(path, 0755) != 0 && errno != EEXIST)) goto out;
	if (pj(path, dir, "pending") || (mkdir(path, 0755) != 0 && errno != EEXIST)) goto out;
	if (pj(path, dir, "miners") || (mkdir(path, 0755) != 0 && errno != EEXIST)) goto out;
	if (pj(path, dir, "genesis.mlf2") || tr_save(m, path, blob, blen, eos)) goto out;
	if (pj(path, dir, "tip.mlf2") || tr_save(m, path, blob, blen, eos)) goto out;
	if (opt) {   /* loaded first, so a file from another model is refused here */
		uint64_t orng;
		if (tr_load_opt(m, opt, &orng) || pj(path, dir, "genesis.opt") || tr_save_opt(m, path, orng)) goto out;
	}
	if (!(text = tr_slurp(data, &tlen)) || !(ids = tr_encode(blob, blen, eos, text, tlen, &n))) goto out;
	if (too_short(n, p->shards, p->stripe, T)) {
		fprintf(stderr, "chain: %ld tokens in %d shards of stripes of %d: a stripe must pass the model's ctx %d, "
		        "and every shard needs one (fewer shards, shorter stripes or more text)\n", n, p->shards, p->stripe, T);
		goto out;
	}
	ids_head ih = { {'H','K','I','D'}, CH_VERSION, (uint64_t)n };
	part ps[2] = { { &ih, sizeof ih }, { ids, (size_t)n * sizeof(int) } };
	unsigned char dh[32], gh[32];
	if (pj(path, dir, "ids.bin") || write_hashed(path, ps, 2, dh)) goto out;
	if (genesis_hash(dir, p, dh, gh)) goto out;
	params_text(p, txt, sizeof txt);
	sha256_hex(dh, hx); sha256_hex(gh, hx2);
	char cfg[1024];
	snprintf(cfg, sizeof cfg, "spelaeum %d\n%sdata %s\ngenesis %s\n", CH_VERSION, txt, hx, hx2);
	part pc = { cfg, strlen(cfg) };
	if (pj(path, dir, "chain.cfg") || write_hashed(path, &pc, 1, NULL)) goto out;
	if (write_tip(dir, 0, gh)) goto out;
	fprintf(stderr, "chain: %s, %ld tokens in %d shards (stripes of %d), %zu params, genesis %.16s\n",
	        dir, n, p->shards, p->stripe, tr_n_params(m), hx2);
	if (p->experts) fprintf(stderr, "chain: each job trains %d of each layer's %d experts, the rest frozen\n", p->experts, tr_cfg(m)->experts);
	if (p->trunk) fprintf(stderr, "chain: every %s block is a trunk round: every job trains the trunk, the experts frozen\n", p->trunk == 1 ? "" : "Nth");
	rc = 0;
out:
	free(text); free(ids); free(blob);
	tr_free(m);
	return rc;
}

/* ── mine ───────────────────────────────────────────────────────────────── */

int ch_mine(const char *dir, const char *miner, char commit_hex[65]) {
	if (!miner_ok(miner)) { fprintf(stderr, "chain: a miner name is 1-31 of a-z A-Z 0-9 - _ .\n"); return -1; }
	ch_config c; ch_ids d = {0};
	uint64_t height; unsigned char parent[32];
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, parent) || ids_open(dir, &d)) { ids_close(&d); return -1; }
	tr_model *m = load_tip(dir, NULL, NULL, NULL);
	int rc = -1, T = m ? tr_cfg(m)->ctx : 0, B = c.p.batch, moe = c.p.experts > 0, nu = m ? job_size(m, &c.p) : -1;
	int trunk = trunk_round(&c.p, height + 1);
	size_t np = m ? tr_n_params(m) : 0;
	float *base = m ? malloc(np * sizeof(float)) : NULL;
	int *in = malloc((size_t)B * T * sizeof(int) + 1), *tg = malloc((size_t)B * T * sizeof(int) + 1);
	uint32_t *un = nu > 0 ? malloc((size_t)nu * sizeof *un) : NULL;
	if (!m || !base || !in || !tg || !un || (nu = job_units(m, &c.p, height + 1, parent, miner, un)) < 1) goto out;
	if (too_short(d.n, c.p.shards, c.p.stripe, T)) { fprintf(stderr, "chain: the data is too short for this model\n"); goto out; }

	/* warm: the miner's own optimizer, else the genesis one, else cold */
	char optp[PATHN], name[96];
	const char *from = "cold";
	uint64_t orng;
	snprintf(name, sizeof name, "miners/%s.opt", miner);
	if (pj(optp, dir, name)) goto out;
	if (access(optp, F_OK) == 0) { if (tr_load_opt(m, optp, &orng)) goto out; from = "its own optimizer"; }
	else {
		char gp[PATHN];
		if (pj(gp, dir, "genesis.opt")) goto out;
		if (access(gp, F_OK) == 0) { if (tr_load_opt(m, gp, &orng)) goto out; from = "the genesis optimizer"; }
	}

	uint32_t shard = ch_shard(parent, miner, c.p.shards);
	unsigned char jh[32];   /* the job's own randomness, as fixed as its shard */
	sha256_ctx sc; sha256_init(&sc);
	sha256_add(&sc, "spelaeum job\n", 13); sha256_add(&sc, parent, 32); sha256_add(&sc, miner, strlen(miner));
	sha256_done(&sc, jh);
	uint64_t rng = h64(jh);
	float *p = tr_params(m);
	memcpy(base, p, np * sizeof(float));
	int warm = c.p.steps / 10 > 0 ? c.p.steps / 10 : 1;
	double loss = 0;
	fprintf(stderr, "mine: height %llu, shard %u of %d, %d steps of %d x %d, %s\n",
	        (unsigned long long)height, shard, c.p.shards, c.p.steps, B, T, from);
	if (trunk) fprintf(stderr, "  a trunk round: the trunk and its routers, every expert frozen\n");
	else if (moe) {   /* its experts and nothing else, on the loss the seal checks: with the routers frozen, aux
	                losses could only bend earlier experts to steer later routers */
		tr_set_aux(m, 0, 0);
		fprintf(stderr, "  experts");
		for (int k = 0; k < nu; k++) fprintf(stderr, " %u.%u", (un[k] - 1) / (uint32_t)tr_cfg(m)->experts, (un[k] - 1) % (uint32_t)tr_cfg(m)->experts);
		fprintf(stderr, " (layer.expert), the rest frozen\n");
	}
	for (int s = 0; s < c.p.steps; s++) {
		float lr = c.p.lr, lo_lr = lr / 10;
		float rate = s < warm ? lr * (s + 1) / warm
		           : lo_lr + 0.5f * (lr - lo_lr) * (1 + cosf(3.14159265f * (s - warm) / (float)(c.p.steps - warm > 0 ? c.p.steps - warm : 1)));
		tr_zero_grad(m);
		for (int b = 0; b < B; b++) {
			long at = ch_window(d.n, c.p.shards, c.p.stripe, (int)shard, mix(rng++), T);
			memcpy(in + (size_t)b * T, d.ids + at, (size_t)T * sizeof(int));
			memcpy(tg + (size_t)b * T, d.ids + at + 1, (size_t)T * sizeof(int));
		}
		double l = tr_run(m, in, tg, B, T, 1.0f / ((float)B * T));
		if (l < 0 || !isfinite(l)) { fprintf(stderr, "mine: the loss went bad at step %d\n", s); goto out; }
		loss = l / ((double)B * T);
		if (moe) outside(m, un, nu, tr_grads(m), NULL);   /* before the clip: frozen grads count for nothing */
		tr_clip(m, 1.0);
		tr_adamw(m, rate, 0.0f, 0.9f, 0.95f, 1e-8f);
		if (moe) { outside(m, un, nu, p, base); (void)tr_params(m); }   /* warm moments move frozen weights: put them back */
		if ((s + 1) % 50 == 0 || s + 1 == c.p.steps) fprintf(stderr, "  step %d/%d  loss %.4f\n", s + 1, c.p.steps, loss);
	}
	size_t nd = 0;   /* base is the delta now, packed unit by unit: the write never passes the read */
	for (int k = 0; k < nu; k++) {
		size_t len, off = unit_at(m, moe, un[k], &len);
		for (size_t j = 0; j < len; j++) base[nd++] = p[off + j] - base[off + j];
	}
	{	/* the optimizer goes home with the miner, whatever the seal makes of the delta */
		char md[PATHN];
		if (pj(md, dir, "miners") || (mkdir(md, 0755) != 0 && errno != EEXIST) || tr_save_opt(m, optp, rng)) goto out;
	}

	ch_dhead h;
	memset(&h, 0, sizeof h);
	memcpy(h.magic, "HKDL", 4); h.version = moe ? CH_UNITS : CH_VERSION;
	memcpy(h.parent, parent, 32);
	h.n = nd; h.shard = shard; h.steps = (uint32_t)c.p.steps; h.train_loss = loss;
	memcpy(h.miner, miner, strlen(miner));
	char path[PATHN];
	snprintf(name, sizeof name, "pending/.mine-%ld", (long)getpid());
	if (pj(path, dir, name)) goto out;
	uint32_t nun = (uint32_t)nu;
	part ps[4] = { { &h, sizeof h }, { &nun, moe ? 4 : 0 }, { un, moe ? (size_t)nu * 4 : 0 }, { base, nd * sizeof(float) } };
	unsigned char commit[32];
	if (write_hashed(path, ps, 4, commit)) goto out;
	char final[PATHN];
	sha256_hex(commit, commit_hex);
	snprintf(name, sizeof name, "pending/%s.dlt", commit_hex);
	if (pj(final, dir, name) || rename(path, final) != 0) { perror(final); remove(path); goto out; }
	rc = 0;
out:
	free(base); free(in); free(tg); free(un);
	tr_free(m); ids_close(&d);
	return rc;
}

/* ── seal ───────────────────────────────────────────────────────────────── */

typedef struct { ch_contrib c; char path[PATHN]; } pend;
static int cmp_pend(const void *a, const void *b) { return memcmp(((const pend *)a)->c.commit, ((const pend *)b)->c.commit, 32); }

static void set_aside(const char *path, const char *why, const char *suffix) {
	char to[PATHN + 16];
	snprintf(to, sizeof to, "%s.%s", path, suffix);
	fprintf(stderr, "  %s: %s, set aside\n", strrchr(path, '/') ? strrchr(path, '/') + 1 : path, why);
	rename(path, to);
}

/* a pending delta checked against the tip: name = hash, shape, shard, units, finite.
 * un gets its units (job_size of them), the floats come back packed in their order */
static float *read_delta(const char *path, const unsigned char want[32], const unsigned char parent[32],
                         const tr_model *m, const ch_params *cp, uint64_t height, uint32_t *un, int *nun, ch_dhead *out, const char **why) {
	size_t len; unsigned char hash[32];
	unsigned char *b = slurp_hash(path, &len, hash);
	*why = "unreadable";
	if (!b) return NULL;
	ch_dhead *h = (ch_dhead *)b;
	int moe = cp->experts > 0, nu = 0;
	size_t at, n;
	if (len < sizeof *h || memcmp(h->magic, "HKDL", 4)) { *why = "not a delta"; goto bad; }
	if (memchr(h->miner, 0, CH_MINER) == NULL || !miner_ok(h->miner)) { *why = "bad miner name"; goto bad; }
	if ((nu = job_units(m, cp, height, parent, h->miner, un)) < 1) { *why = "no units for it"; goto bad; }
	at = sizeof *h + (moe ? 4 + 4 * (size_t)nu : 0); n = units_len(m, moe, un, nu);
	if (h->version != (moe ? CH_UNITS : CH_VERSION)) { *why = moe ? "a dense delta on an expert chain" : "an expert delta on a dense chain"; goto bad; }
	if (memcmp(hash, want, 32)) { *why = "its name is not its hash"; goto bad; }
	if (memcmp(h->parent, parent, 32)) { *why = "stale (mined on another block)"; goto bad; }
	if (h->n != n || len != at + n * 4) { *why = "wrong size for this model"; goto bad; }
	if (h->shard != ch_shard(parent, h->miner, cp->shards)) { *why = "trained on a shard it was not given"; goto bad; }
	if (moe && (len < at || memcmp(b + sizeof *h, &(uint32_t){ (uint32_t)nu }, 4) || memcmp(b + sizeof *h + 4, un, 4 * (size_t)nu))) {
		*why = "it names units it was not given"; goto bad;
	}
	float *dl = (float *)(b + at);
	for (size_t j = 0; j < n; j++) if (!isfinite(dl[j])) { *why = "not finite"; goto bad; }
	*out = *h; *nun = nu;
	memmove(b, dl, n * 4);   /* hand back the floats at the start of the buffer */
	return (float *)b;
bad:
	free(b);
	return NULL;
}

int ch_seal(const char *dir) {
	ch_config c; ch_ids d = {0};
	uint64_t height; unsigned char parent[32];
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, parent) || ids_open(dir, &d)) { ids_close(&d); return -1; }
	double diff;
	if (next_difficulty(dir, &c, height, &diff)) { ids_close(&d); return -1; }
	void *blob = NULL; size_t blen = 0; int eos = 0;
	tr_model *m = load_tip(dir, &blob, &blen, &eos);
	int rc = -1, T = m ? tr_cfg(m)->ctx : 0, B = c.p.batch, np_ = 0, cap = 0, moe = c.p.experts > 0, nu = m ? job_size(m, &c.p) : -1;
	int nall = moe && m ? 1 + tr_cfg(m)->layers * tr_cfg(m)->experts : 1;   /* units in the model */
	size_t np = m ? tr_n_params(m) : 0;
	pend *ps = NULL;
	float *base = NULL, *sum = NULL, *use = NULL, *keep = NULL;
	long *at = NULL;
	uint32_t *skip = NULL, *un = NULL, *ub = NULL, *um = NULL, *bu = NULL;   /* a delta's, the best's, the merge's, the block's */
	unsigned char (*commits)[32] = NULL;
	int *in = NULL, *tg = NULL, *cnt = NULL, nb = 0;
	DIR *dd = NULL;
	char path[PATHN];
	if (!m || nu < 0) goto out;
	un = malloc((size_t)nu * sizeof *un); ub = malloc((size_t)nu * sizeof *ub);
	um = malloc((size_t)nall * sizeof *um); cnt = calloc((size_t)nall, sizeof *cnt);
	if (!un || !ub || !um || !cnt) goto out;

	/* the pending deltas, checked; dupes of a miner past its first set aside */
	if (pj(path, dir, "pending") || !(dd = opendir(path))) { perror(path); goto out; }
	struct dirent *e;
	while ((e = readdir(dd))) {
		size_t nl = strlen(e->d_name);
		if (nl != 68 || strcmp(e->d_name + 64, ".dlt")) continue;
		if (np_ == cap) {
			cap = cap ? cap * 2 : 8;
			pend *nb = realloc(ps, (size_t)cap * sizeof *ps);
			if (!nb) goto out;
			ps = nb;
		}
		pend *q = &ps[np_];
		memset(q, 0, sizeof *q);
		char name[96];
		snprintf(name, sizeof name, "pending/%s", e->d_name);
		if (pj(q->path, dir, name) || unhex(e->d_name, q->c.commit)) continue;
		ch_dhead h; const char *why;
		float *dl = read_delta(q->path, q->c.commit, parent, m, &c.p, height + 1, un, &nu, &h, &why);
		if (!dl) { set_aside(q->path, why, strstr(why, "stale") ? "stale" : "bad"); continue; }
		free(dl);
		memcpy(q->c.miner, h.miner, CH_MINER); q->c.shard = h.shard;
		np_++;
	}
	closedir(dd); dd = NULL;
	qsort(ps, (size_t)np_, sizeof *ps, cmp_pend);
	int k = 0;
	for (int i = 0; i < np_; i++) {
		int dup = 0;
		for (int j = 0; j < k; j++) dup |= !strcmp(ps[j].c.miner, ps[i].c.miner);
		if (dup) set_aside(ps[i].path, "a second delta from the same miner this round", "dup");
		else ps[k++] = ps[i];
	}
	if (!k) { fprintf(stderr, "seal: nothing pending above height %llu\n", (unsigned long long)height); rc = 1; goto out; }

	/* the seed from every commit, then the windows it picks */
	commits = malloc((size_t)k * 32); skip = malloc((size_t)k * sizeof *skip);
	at = malloc((size_t)c.p.eval_windows * sizeof *at);
	base = malloc(np * sizeof(float)); sum = calloc(np, sizeof(float));
	in = malloc((size_t)B * T * sizeof(int)); tg = malloc((size_t)B * T * sizeof(int));
	if (!commits || !skip || !at || !base || !sum || !in || !tg) goto out;
	for (int i = 0; i < k; i++) { memcpy(commits[i], ps[i].c.commit, 32); skip[i] = ps[i].c.shard; }
	unsigned char seed[32];
	ch_seed(parent, (const unsigned char (*)[32])commits, k, seed);
	if (pick_windows(seed, &d, c.p.shards, c.p.stripe, skip, k, T, c.p.eval_windows, at)) goto out;

	float *p = tr_params(m);
	memcpy(base, p, np * sizeof(float));
	double pl = eval_loss(m, d.ids, at, c.p.eval_windows, B, T, in, tg);
	if (pl < 0) goto out;
	fprintf(stderr, "seal: height %llu -> %llu, %d deltas, difficulty %.5f, parent loss %.4f on %d windows\n",
	        (unsigned long long)height, (unsigned long long)height + 1, k, diff, pl, c.p.eval_windows);
	int passed = 0, best = -1;
	for (int i = 0; i < k; i++) {
		ch_dhead h; const char *why;
		float *dl = read_delta(ps[i].path, ps[i].c.commit, parent, m, &c.p, height + 1, un, &nu, &h, &why);
		if (!dl) goto out;
		apply(m, moe, un, nu, p, base, dl);
		double l = eval_loss(m, d.ids, at, c.p.eval_windows, B, T, in, tg);
		apply(m, moe, un, nu, p, base, NULL);
		if (l < 0) { free(dl); goto out; }
		ps[i].c.loss = l;
		ps[i].c.state = pl - l >= diff ? CH_PASSED : CH_REJECTED;
		if (ps[i].c.state == CH_PASSED) {
			apply(m, moe, un, nu, sum, NULL, dl);
			for (int q = 0; q < nu; q++) cnt[un[q]]++;
			passed++;
			if (best < 0 || l < ps[best].c.loss) best = i;
		}
		fprintf(stderr, "  %-20s shard %3u  loss %.4f  cut %+.4f  %s\n", ps[i].c.miner, ps[i].c.shard, l, pl - l,
		        ps[i].c.state == CH_PASSED ? "passes" : "short of the difficulty");
		free(dl);
	}

	/* the block's delta: per unit, the mean of the passes that carry it (one carrier: its
	 * floats as they are), when that beats the best one alone */
	double nl = pl;
	size_t nd = 0;
	if (passed) {
		ch_dhead h; const char *why;
		keep = read_delta(ps[best].path, ps[best].c.commit, parent, m, &c.p, height + 1, ub, &nb, &h, &why);
		if (!keep) goto out;
		use = keep; nl = ps[best].c.loss; bu = ub;
		if (passed > 1) {
			int nm = 0;
			size_t w = 0, len;
			for (int u = 0; u < nall; u++) {   /* packed forward in place: w never passes off */
				if (!cnt[u]) continue;
				size_t off = unit_at(m, moe, (uint32_t)u, &len);
				for (size_t j = 0; j < len; j++) sum[w + j] = sum[off + j] / (float)cnt[u];
				um[nm++] = (uint32_t)u; w += len;
			}
			apply(m, moe, um, nm, p, base, sum);
			double ml = eval_loss(m, d.ids, at, c.p.eval_windows, B, T, in, tg);
			apply(m, moe, um, nm, p, base, NULL);
			if (ml < 0) goto out;
			fprintf(stderr, "  merged %d: loss %.4f against the best alone %.4f\n", passed, ml, nl);
			if (ml <= nl) { use = sum; nl = ml; bu = um; nb = nm; }
		}
		for (int i = 0; i < k; i++)
			if (ps[i].c.state == CH_PASSED && (use == sum || i == best)) ps[i].c.state = CH_MERGED;
		nd = apply(m, moe, bu, nb, p, base, use);
	}

	ch_bhead bh;
	memset(&bh, 0, sizeof bh);
	memcpy(bh.magic, "HKBK", 4); bh.version = moe ? CH_UNITS : CH_VERSION;
	bh.height = height + 1;
	memcpy(bh.parent, parent, 32); memcpy(bh.seed, seed, 32);
	bh.difficulty = diff; bh.parent_loss = pl; bh.new_loss = nl;
	bh.ncontrib = (uint32_t)k; bh.n = nd;
	ch_contrib *cs = malloc((size_t)k * sizeof *cs);
	if (!cs) goto out;
	for (int i = 0; i < k; i++) { cs[i] = ps[i].c; bh.nmerged += cs[i].state == CH_MERGED; }
	uint32_t nbu = (uint32_t)nb;
	part bp[5] = { { &bh, sizeof bh }, { cs, (size_t)k * sizeof *cs }, { &nbu, moe ? 4 : 0 }, { bu, moe ? (size_t)nb * 4 : 0 },
	               { use, nd * sizeof(float) } };
	unsigned char bhash[32];
	int wr = block_path(dir, bh.height, path) || write_hashed(path, bp, 5, bhash);
	free(cs);
	if (wr) goto out;
	if (pj(path, dir, "tip.mlf2") || tr_save(m, path, blob, blen, eos) || write_tip(dir, bh.height, bhash)) goto out;
	for (int i = 0; i < k; i++) remove(ps[i].path);
	char hx[65];
	sha256_hex(bhash, hx);
	fprintf(stderr, "seal: block %llu %.16s, %u of %d merged, loss %.4f -> %.4f, next difficulty %.5f\n",
	        (unsigned long long)bh.height, hx, bh.nmerged, k, pl, nl, ch_retarget(diff, k, passed));
	rc = 0;
out:
	if (dd) closedir(dd);
	free(ps); free(base); free(sum); free(keep); free(at); free(skip); free(commits); free(in); free(tg); free(blob);
	free(un); free(ub); free(um); free(cnt);
	tr_free(m); ids_close(&d);
	return rc;
}

/* ── verify ─────────────────────────────────────────────────────────────── */

static int close_enough(double a, double b) { return fabs(a - b) <= 1e-5 * (1 + fabs(b)); }

int ch_verify(const char *dir, int full) {
	ch_config c; ch_ids d = {0};
	uint64_t tip_h; unsigned char tip_hash[32];
	if (ch_load_config(dir, &c) || ch_tip(dir, &tip_h, tip_hash) || ids_open(dir, &d)) { ids_close(&d); return -1; }
	char path[PATHN];
	int rc = -1;
	tr_model *m = NULL, *t = NULL;
	long *at = NULL;
	int *in = NULL, *tg = NULL;
	uint32_t *un = NULL;
	unsigned char dh[32], gh[32], prev[32], *mark = NULL;
	sha256(d.map, d.size, dh);
	if (memcmp(dh, c.data, 32)) { fprintf(stderr, "verify: ids.bin is not the data this chain was made on\n"); goto out; }
	if (genesis_hash(dir, &c.p, dh, gh) || memcmp(gh, c.genesis, 32)) { fprintf(stderr, "verify: genesis.mlf2 or chain.cfg changed\n"); goto out; }
	if (pj(path, dir, "genesis.mlf2") || !(m = tr_load(path, NULL, NULL, NULL))) goto out;
	int T = tr_cfg(m)->ctx, B = c.p.batch, moe = c.p.experts > 0, nu = job_size(m, &c.p);
	int nall = moe ? 1 + tr_cfg(m)->layers * tr_cfg(m)->experts : 1;
	size_t np = tr_n_params(m);
	float *p = tr_params(m);
	if (nu < 0) goto out;
	at = malloc((size_t)c.p.eval_windows * sizeof *at);
	in = malloc((size_t)B * T * sizeof(int)); tg = malloc((size_t)B * T * sizeof(int));
	un = malloc((size_t)nu * sizeof *un); mark = malloc((size_t)nall);
	if (!at || !in || !tg || !un || !mark) goto out;
	memcpy(prev, gh, 32);
	double diff = c.p.difficulty;
	uint64_t h = 0;
	for (;;) {
		if (block_path(dir, h + 1, path)) goto out;
		if (access(path, F_OK) != 0) break;
		h++;
		unsigned char *b; ch_bhead *hd; ch_contrib *cs; const float *dl; const uint32_t *bu; uint32_t nb; unsigned char bhash[32];
		if (read_block(path, moe, &b, &hd, &cs, &bu, &nb, &dl, bhash)) goto out;
		static const uint32_t whole = 0;   /* a dense block's one unit, when it carries a delta */
		if (!moe) { bu = &whole; nb = hd->n ? 1 : 0; }
		const char *bad = NULL;
		unsigned char (*cm)[32] = malloc((size_t)hd->ncontrib * 32 + 1);
		uint32_t *skip = malloc((size_t)hd->ncontrib * sizeof *skip + 1);
		unsigned char seed[32];
		int passed = 0, merged = 0;
		if (!cm || !skip) bad = "out of memory";
		else if (hd->height != h) bad = "wrong height";
		else if (memcmp(hd->parent, prev, 32)) bad = "its parent is not the block before it";
		else if (hd->difficulty != diff) bad = "wrong difficulty";
		else if (!moe && hd->n != 0 && hd->n != np) bad = "wrong delta size";
		for (uint32_t i = 0; !bad && i < hd->ncontrib; i++) {
			const ch_contrib *x = &cs[i];
			if (i && memcmp(cs[i - 1].commit, x->commit, 32) >= 0) bad = "deltas out of commit order";
			else if (!memchr(x->miner, 0, CH_MINER) || !miner_ok(x->miner)) bad = "bad miner name";
			else if (x->shard != ch_shard(prev, x->miner, c.p.shards)) bad = "a delta on a shard it was not given";
			else if (x->state > CH_MERGED || (x->state != CH_REJECTED) != (hd->parent_loss - x->loss >= hd->difficulty))
				bad = "a delta's state does not match its loss";
			memcpy(cm[i], x->commit, 32); skip[i] = x->shard;
			passed += x->state != CH_REJECTED; merged += x->state == CH_MERGED;
		}
		if (!bad && (merged != (int)hd->nmerged || (merged == 0) != (hd->n == 0))) bad = "merged count does not match the delta";
		if (!bad && moe) {   /* its units: the union of its merged miners', in order (so none past the end, the
		                        trunk in a trunk round only), and exactly their floats */
			memset(mark, 0, (size_t)nall);
			for (uint32_t i = 0; i < hd->ncontrib; i++)
				if (cs[i].state == CH_MERGED && (nu = job_units(m, &c.p, h, prev, cs[i].miner, un)) > 0)
					for (int q = 0; q < nu; q++) mark[un[q]] = 1;
			uint32_t q = 0;
			int same = 1;
			for (int u = 0; u < nall; u++) if (mark[u]) same &= q < nb && bu[q++] == (uint32_t)u;
			if (!same || q != nb) bad = "its units are not the ones its merged miners were given";
			else if (hd->n != units_len(m, moe, bu, (int)nb)) bad = "wrong delta size";
		}
		if (!bad) {
			ch_seed(prev, (const unsigned char (*)[32])cm, (int)hd->ncontrib, seed);
			if (memcmp(seed, hd->seed, 32)) bad = "wrong seed";
		}
		if (!bad && full) {
			if (pick_windows(seed, &d, c.p.shards, c.p.stripe, skip, (int)hd->ncontrib, T, c.p.eval_windows, at)) bad = "no windows";
			else {
				double pl = eval_loss(m, d.ids, at, c.p.eval_windows, B, T, in, tg);
				if (!close_enough(pl, hd->parent_loss)) bad = "parent loss does not reproduce";
			}
		}
		if (!bad) apply(m, moe, bu, (int)nb, p, NULL, dl);
		if (!bad && full && hd->n) {
			double nl = eval_loss(m, d.ids, at, c.p.eval_windows, B, T, in, tg);
			if (!close_enough(nl, hd->new_loss)) bad = "the block's loss does not reproduce";
		}
		if (!bad) diff = ch_retarget(hd->difficulty, (int)hd->ncontrib, passed);
		free(cm); free(skip); free(b);
		if (bad) { fprintf(stderr, "verify: block %llu: %s\n", (unsigned long long)h, bad); goto out; }
		memcpy(prev, bhash, 32);
	}
	if (h != tip_h || memcmp(prev, tip_hash, 32)) { fprintf(stderr, "verify: tip says %llu, the blocks end at %llu or hash elsewhere\n", (unsigned long long)tip_h, (unsigned long long)h); goto out; }
	if (pj(path, dir, "tip.mlf2") || !(t = tr_load(path, NULL, NULL, NULL)) || tr_n_params(t) != np ||
	    memcmp(tr_params(t), p, np * sizeof(float))) { fprintf(stderr, "verify: tip.mlf2 is not the replayed model\n"); goto out; }
	char hx[65];
	sha256_hex(prev, hx);
	fprintf(stderr, "verify: the chain holds, %llu blocks, tip %.16s%s\n", (unsigned long long)h, hx, full ? ", every loss reproduced" : "");
	rc = 0;
out:
	free(at); free(in); free(tg); free(un); free(mark);
	tr_free(m); tr_free(t); ids_close(&d);
	return rc;
}
