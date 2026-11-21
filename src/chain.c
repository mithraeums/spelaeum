/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* chain.c — Spelaeum: many machines training one model, a block at a time. See chain.h for the shape. */
#include "chain.h"
#include "ed25519.h"
#include "iloss.h"
#include "sha256.h"
#include "train.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(sizeof(ch_dhead) == 128, "ch_dhead is written as is");
_Static_assert(sizeof(ch_bhead) == 184, "ch_bhead is written as is");
_Static_assert(sizeof(ch_contrib) == 176, "ch_contrib is written as is");
_Static_assert(sizeof(ch_verdict) == 136, "ch_verdict is sent and kept as is");
/* every file and message is these bytes on every machine: little-endian, IEEE floats, and the
   token positions 64 bits wide where C's long is 32 (wasm32, 32-bit phones) */
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "chain files are little-endian");
_Static_assert(sizeof(float) == 4 && FLT_MANT_DIG == 24 && sizeof(double) == 8 && DBL_MANT_DIG == 53, "IEEE floats");

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
static int name_ok(const char *s) {
	size_t n = strnlen(s, CH_NAME);
	if (n == 0 || n >= CH_NAME) return 0;
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
/* the training tokens, read a window at a time and never held whole: in a browser
 * there is no mmap, and a tab on a phone has no room for the file twice */
typedef struct { int fd, open; int64_t n; ids_head h; } ch_ids;

static int pread_all(int fd, void *buf, size_t len, off_t off) {
	char *p = buf;
	while (len) {
		ssize_t r = pread(fd, p, len, off);
		if (r < 0 && errno == EINTR) continue;
		if (r <= 0) return -1;
		p += r; len -= (size_t)r; off += r;
	}
	return 0;
}
/* n tokens from token `at` */
static int ids_read(const ch_ids *d, int64_t at, int64_t n, int *out) {
	if (!d->open || at < 0 || n < 0 || at > d->n - n) return -1;
	return pread_all(d->fd, out, (size_t)n * sizeof(int), (off_t)(sizeof(ids_head) + (uint64_t)at * sizeof(int)));
}

/* The data's hash: sha256 of "spelaeum data\n", the head, then the sha256 of every
 * stripe of `stripe` tokens in order (the last may be short). Those bytes are what
 * a stripe index would hold, so a trainer can check one stripe it fetched alone
 * against the chain, without the whole file. From the tokens in memory (ids), or
 * read from d a run of stripes at a time. */
static int data_hash(const ids_head *h, const int *ids, const ch_ids *d, int stripe, unsigned char out[32]) {
	enum { RUN = 64 };
	int *buf = ids ? NULL : malloc((size_t)stripe * RUN * sizeof(int));
	if (!ids && !buf) return -1;
	sha256_ctx c; sha256_init(&c);
	sha256_add(&c, "spelaeum data\n", 14);
	sha256_add(&c, h, sizeof *h);
	for (uint64_t at = 0; at < h->n; at += (uint64_t)stripe * RUN) {
		uint64_t run = h->n - at < (uint64_t)stripe * RUN ? h->n - at : (uint64_t)stripe * RUN;
		const int *p = ids ? ids + at : buf;
		if (!ids && ids_read(d, (int64_t)at, (int64_t)run, buf)) { free(buf); return -1; }
		for (uint64_t o = 0; o < run; o += (uint64_t)stripe) {
			uint64_t k = run - o < (uint64_t)stripe ? run - o : (uint64_t)stripe;
			unsigned char sh[32];
			sha256(p + o, (size_t)k * sizeof(int), sh);
			sha256_add(&c, sh, 32);
		}
	}
	sha256_done(&c, out);
	free(buf);
	return 0;
}

static int ids_open(const char *dir, ch_ids *d) {
	char path[PATHN];
	if (pj(path, dir, "ids.bin")) return -1;
	int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st) != 0) { perror(path); if (fd >= 0) close(fd); return -1; }
	ids_head *h = &d->h;
	uint64_t size = (uint64_t)st.st_size;
	if (size < sizeof *h || pread_all(fd, h, sizeof *h, 0) || memcmp(h->magic, "HKID", 4) || h->version != CH_VERSION ||
	    h->n > (size - sizeof *h) / 4 || sizeof *h + h->n * 4 != size) {
		fprintf(stderr, "chain: %s is damaged\n", path);
		close(fd); return -1;
	}
	d->fd = fd; d->open = 1;
	d->n = (int64_t)h->n;
	return 0;
}
static void ids_close(ch_ids *d) { if (d->open) close(d->fd); d->open = 0; }

/* ── the pieces both sides compute ──────────────────────────────────────── */

uint32_t ch_shard(const unsigned char parent[32], const unsigned char key[32], int shards) {
	unsigned char h[32];
	sha256_ctx c; sha256_init(&c);
	sha256_add(&c, parent, 32); sha256_add(&c, key, 32);
	sha256_done(&c, h);
	return (uint32_t)(h64(h) % (uint64_t)shards);
}

/* Floyd's sampling per layer: per draws from the layer's own hash, each a new
 * expert, kept sorted as they land */
int ch_units(const unsigned char parent[32], const unsigned char key[32], int layers, int experts, int per, uint32_t *out) {
	if (layers < 1 || per < 1 || per > experts) return -1;
	for (int l = 0; l < layers; l++) {
		unsigned char h[32], lb[4] = { (unsigned char)l, (unsigned char)(l >> 8), (unsigned char)(l >> 16), (unsigned char)(l >> 24) };
		sha256_ctx c; sha256_init(&c);
		sha256_add(&c, "spelaeum units\n", 15); sha256_add(&c, parent, 32);
		sha256_add(&c, key, 32); sha256_add(&c, lb, 4);
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
static int job_units(const tr_model *m, const ch_params *cp, uint64_t h, const unsigned char parent[32], const unsigned char key[32], uint32_t *un) {
	if (!cp->experts || trunk_round(cp, h)) { un[0] = 0; return 1; }
	return ch_units(parent, key, tr_cfg(m)->layers, tr_cfg(m)->experts, cp->experts, un);
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
/* sha256 of every weight as a float: what a block head says its model is */
static void model_hash(const tr_model *m, unsigned char out[32]) { sha256(tr_params((tr_model *)m), tr_n_params(m) * sizeof(float), out); }

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

int64_t ch_window(int64_t ntok, int shards, int stripe, int s, uint64_t r, int T) {
	int64_t per = ntok / ((int64_t)stripe * shards);   /* stripes each shard owns */
	int64_t k = (int64_t)(r % (uint64_t)per) * shards + s;
	return k * stripe + (int64_t)(mix(r ^ 0x5851F42D4C957F2DULL) % (uint64_t)(stripe - T - 1));
}

/* a corpus too short for every shard to own a stripe a window fits in */
static int too_short(int64_t ntok, int shards, int stripe, int T) {
	return stripe < T + 2 || ntok / ((int64_t)stripe * shards) < 1;
}

/* the round's windows: n starts of T+1 tokens, from shards nobody trained this round
 * (every shard when the round covered them all) */
static int pick_windows(const unsigned char seed[32], const ch_ids *d, int shards, int stripe,
                        const uint32_t *skip, int nskip, int T, int n, int64_t *at) {
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

/* nats per token over the windows, in integers (iloss.h): the same double on every
 * machine for the same model, so a check is a match, not a tolerance. -1 when out
 * of memory or a window is bad. The callers change the weights through a pointer
 * they already hold; il_new reads them through tr_params, which drops the
 * trainer's transposed copies first. */
static double eval_loss(tr_model *m, const ch_ids *d, const int64_t *at, int n, int T) {
	int *w = malloc((size_t)n * (T + 1) * sizeof(int) + 1);   /* the windows side by side */
	int64_t *pos = malloc((size_t)n * sizeof *pos + 1), nll = 0;
	il_model *I = NULL;
	int bad = !w || !pos;
	for (int i = 0; !bad && i < n; i++) { pos[i] = (int64_t)i * (T + 1); bad = ids_read(d, at[i], T + 1, w + pos[i]); }
	if (!bad) { I = il_new(m); bad = !I || il_nll(I, w, pos, n, T, &nll); }
	il_free(I); free(w); free(pos);
	return bad ? -1 : il_nats(nll, (int64_t)n * T);
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

/* the bytes of a block's head part: the head, the contributions, an expert chain's units */
static size_t head_len(const ch_bhead *hd, int moe, uint32_t nu) {
	return sizeof *hd + (size_t)hd->ncontrib * sizeof(ch_contrib) + (moe ? 4 + 4 * (size_t)nu : 0);
}

/* a block file, checked for shape. Always its head part, hash = sha256 of that part;
 * the floats too when delta is not NULL, checked against the head's body hash, or
 * *delta = NULL when the file was pruned to its head part. *buf owns what was read;
 * *hlen, when not NULL, the head part's length. */
static int read_block_at(const char *path, int moe, unsigned char **buf, ch_bhead **h, ch_contrib **cs,
                         const uint32_t **un, uint32_t *nu, const float **delta, unsigned char hash[32], size_t *hlen) {
	FILE *f = fopen(path, "rb");
	if (!f) { fprintf(stderr, "chain: cannot read %s\n", path); return -1; }
	struct stat st;
	ch_bhead hd;
	uint32_t n = 0;
	size_t len = 0, at = 0, head = 0;
	unsigned char *b = NULL;
	int ok = fstat(fileno(f), &st) == 0 && st.st_size >= 0 && fread(&hd, sizeof hd, 1, f) == 1;
	if (ok) len = (size_t)st.st_size;
	ok = ok && !memcmp(hd.magic, "HKBK", 4) && hd.version == (moe ? CH_UNITS : CH_VERSION) && hd.ncontrib <= 1u << 20;
	if (ok) { at = sizeof hd + (size_t)hd.ncontrib * sizeof(ch_contrib); ok = len >= at + (moe ? 4 : 0); }
	if (ok && moe) ok = fseek(f, (long)at, SEEK_SET) == 0 && fread(&n, 4, 1, f) == 1 && n <= (len - at - 4) / 4;
	if (ok) head = head_len(&hd, moe, n);
	int whole = ok && hd.n <= (len - head) / 4 && len == head + hd.n * 4;
	ok = ok && (whole || len == head);   /* whole, or pruned to its head part */
	size_t want = delta && whole ? len : head;
	if (ok && (!(b = malloc(want + 1)) || fseek(f, 0, SEEK_SET) != 0 || fread(b, 1, want, f) != want)) ok = 0;
	fclose(f);
	if (ok) {
		sha256(b, head, hash);
		if (delta && whole) {
			unsigned char bh[32];
			sha256(b + head, len - head, bh);
			ok = !memcmp(bh, hd.body, 32);
		}
	}
	if (!ok) { fprintf(stderr, "chain: %s is damaged\n", path); free(b); return -1; }
	*buf = b; *h = (ch_bhead *)b;
	*cs = (ch_contrib *)(b + sizeof hd);
	*un = (const uint32_t *)(b + at + (moe ? 4 : 0)); *nu = n;
	if (delta) *delta = whole ? (const float *)(b + head) : NULL;
	if (hlen) *hlen = head;
	return 0;
}
static int read_block(const char *path, int moe, unsigned char **buf, ch_bhead **h, ch_contrib **cs,
                      const uint32_t **un, uint32_t *nu, const float **delta, unsigned char hash[32]) {
	return read_block_at(path, moe, buf, h, cs, un, nu, delta, hash, NULL);
}

/* the difficulty block h+1 must meet, from block h (or the config at genesis) */
/* the last block before b of b's kind (trunk round or not), 0 when b is the first:
 * each kind keeps its own difficulty, since a trunk job and an expert job cut the
 * loss by different amounts and one bar would shut the smaller one out */
static uint64_t kind_before(const ch_params *cp, uint64_t b) {
	if (!(cp->experts > 0 && cp->trunk > 0)) return b - 1;
	uint64_t n = (uint64_t)cp->trunk;
	if (b % n == 0) return b - n;
	return b > 1 && (b - 1) % n == 0 ? b - 2 : b - 1;   /* trunk >= 2 here: never two trunk rounds in a row */
}
/* the difficulty block h+1 must meet: retargeted from the last block of its kind, or the config's */
static int next_difficulty(const char *dir, const ch_config *c, uint64_t h, double *d) {
	h = kind_before(&c->p, h + 1);
	if (h == 0) { *d = c->p.difficulty; return 0; }
	char path[PATHN];
	unsigned char *b; ch_bhead *hd; ch_contrib *cs; const uint32_t *bu; uint32_t nb; unsigned char hash[32];
	if (block_path(dir, h, path) || read_block(path, c->p.experts > 0, &b, &hd, &cs, &bu, &nb, NULL, hash)) return -1;
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
	int64_t n = 0;
	char *text = NULL;
	int *ids = NULL;
	if (!m || job_size(m, p) < 0) goto out;
	if (mkdir(dir, 0755) != 0 && errno != EEXIST) { perror(dir); goto out; }
	if (pj(path, dir, "chain.cfg")) goto out;
	if (access(path, F_OK) == 0) { fprintf(stderr, "chain: %s is a chain already\n", dir); goto out; }
	if (pj(path, dir, "blocks") || (mkdir(path, 0755) != 0 && errno != EEXIST)) goto out;
	if (pj(path, dir, "pending") || (mkdir(path, 0755) != 0 && errno != EEXIST)) goto out;
	if (pj(path, dir, "trainers") || (mkdir(path, 0755) != 0 && errno != EEXIST)) goto out;
	if (pj(path, dir, "genesis.mlf2") || tr_save(m, path, blob, blen, eos)) goto out;
	if (pj(path, dir, "tip.mlf2") || tr_save(m, path, blob, blen, eos)) goto out;
	if (opt) {   /* loaded first, so a file from another model is refused here */
		uint64_t orng;
		if (tr_load_opt(m, opt, &orng) || pj(path, dir, "genesis.opt") || tr_save_opt(m, path, orng)) goto out;
	}
	if (!(text = tr_slurp(data, &tlen)) || !(ids = tr_encode(blob, blen, eos, text, tlen, &n))) goto out;
	if (too_short(n, p->shards, p->stripe, T)) {
		fprintf(stderr, "chain: %" PRId64 " tokens in %d shards of stripes of %d: a stripe must pass the model's ctx %d, "
		        "and every shard needs one (fewer shards, shorter stripes or more text)\n", n, p->shards, p->stripe, T);
		goto out;
	}
	ids_head ih = { {'H','K','I','D'}, CH_VERSION, (uint64_t)n };
	part ps[2] = { { &ih, sizeof ih }, { ids, (size_t)n * sizeof(int) } };
	unsigned char dh[32], gh[32];
	if (pj(path, dir, "ids.bin") || write_hashed(path, ps, 2, dh)) goto out;
	if (data_hash(&ih, ids, NULL, p->stripe, dh)) goto out;
	if (genesis_hash(dir, p, dh, gh)) goto out;
	params_text(p, txt, sizeof txt);
	sha256_hex(dh, hx); sha256_hex(gh, hx2);
	char cfg[1024];
	snprintf(cfg, sizeof cfg, "spelaeum %d\n%sdata %s\ngenesis %s\n", CH_VERSION, txt, hx, hx2);
	part pc = { cfg, strlen(cfg) };
	if (pj(path, dir, "chain.cfg") || write_hashed(path, &pc, 1, NULL)) goto out;
	if (write_tip(dir, 0, gh)) goto out;
	fprintf(stderr, "chain: %s, %" PRId64 " tokens in %d shards (stripes of %d), %zu params, genesis %.16s\n",
	        dir, n, p->shards, p->stripe, tr_n_params(m), hx2);
	if (p->experts) fprintf(stderr, "chain: each job trains %d of each layer's %d experts, the rest frozen\n", p->experts, tr_cfg(m)->experts);
	if (p->trunk) fprintf(stderr, "chain: every %s block is a trunk round: every job trains the trunk, the experts frozen\n", p->trunk == 1 ? "" : "Nth");
	rc = 0;
out:
	free(text); free(ids); free(blob);
	tr_free(m);
	return rc;
}

/* ── keys ───────────────────────────────────────────────────────────────── */

/* what a key signs for a delta: the parent it was trained on, its commit, and the name
 * it goes by (all CH_NAME bytes), so a block cannot relabel it */
#define SIGNED (15 + 32 + 32 + CH_NAME)
static void signed_msg(const unsigned char parent[32], const unsigned char commit[32], const char name[CH_NAME], unsigned char out[SIGNED]) {
	memcpy(out, "spelaeum delta\n", 15); memcpy(out + 15, parent, 32); memcpy(out + 47, commit, 32); memcpy(out + 79, name, CH_NAME);
}
static void sign_commit(const ch_key *k, const unsigned char parent[32], const unsigned char commit[32], const char name[CH_NAME], unsigned char sig[CH_SIG]) {
	unsigned char msg[SIGNED];
	signed_msg(parent, commit, name, msg);
	ed25519_sign(sig, msg, sizeof msg, k->seed, k->pk);
}
static int sig_ok(const unsigned char key[32], const unsigned char parent[32], const unsigned char commit[32], const char name[CH_NAME], const unsigned char sig[CH_SIG]) {
	unsigned char msg[SIGNED];
	signed_msg(parent, commit, name, msg);
	return ed25519_verify(sig, msg, sizeof msg, key) == 0;
}

void ch_key_from(const unsigned char seed[32], ch_key *k) {
	memcpy(k->seed, seed, 32);
	ed25519_public(seed, k->pk);
}

int ch_key_load(const char *path, ch_key *k, int make) {
	char buf[96];
	unsigned char seed[32];
	FILE *f = fopen(path, "r");
	if (f) {
		size_t n = fread(buf, 1, sizeof buf - 1, f);
		fclose(f);
		buf[n] = 0;
		if (strncmp(buf, "spelaeum key\n", 13) || n < 13 + 64 || unhex(buf + 13, seed)) {
			fprintf(stderr, "chain: %s is not a spelaeum key\n", path); return -1;
		}
		ch_key_from(seed, k);
		memset(seed, 0, sizeof seed); memset(buf, 0, sizeof buf);
		return 0;
	}
	if (errno != ENOENT || !make) { perror(path); return -1; }
	char d[PATHN];   /* its directory, private, when there is one to make */
	snprintf(d, sizeof d, "%s", path);
	char *sl = strrchr(d, '/');
	if (sl && sl != d) { *sl = 0; if (mkdir(d, 0700) != 0 && errno != EEXIST) { perror(d); return -1; } }
	FILE *r = fopen("/dev/urandom", "rb");
	int got = r && fread(seed, 1, 32, r) == 32;
	if (r) fclose(r);
	if (!got) { fprintf(stderr, "chain: no randomness for a key\n"); return -1; }
	int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
	if (fd < 0) { perror(path); return -1; }
	char hx[65];
	sha256_hex(seed, hx);   /* hex of 32 bytes, nothing hashed */
	int len = snprintf(buf, sizeof buf, "spelaeum key\n%s\n", hx);
	int ok = write(fd, buf, (size_t)len) == len;
	if (close(fd) != 0) ok = 0;
	memset(buf, 0, sizeof buf); memset(hx, 0, sizeof hx);
	if (!ok) { fprintf(stderr, "chain: could not write %s\n", path); remove(path); return -1; }
	ch_key_from(seed, k);
	memset(seed, 0, sizeof seed);
	return 0;
}

/* ── train ───────────────────────────────────────────────────────────────── */

int ch_train(const char *dir, const char *trainer, const ch_key *key, char commit_hex[65]) {
	if (!name_ok(trainer)) { fprintf(stderr, "chain: a trainer name is 1-31 of a-z A-Z 0-9 - _ .\n"); return -1; }
	ch_config c; ch_ids d = {0};
	uint64_t height; unsigned char parent[32];
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, parent) || ids_open(dir, &d)) { ids_close(&d); return -1; }
	tr_model *m = load_tip(dir, NULL, NULL, NULL);
	int rc = -1, T = m ? tr_cfg(m)->ctx : 0, B = c.p.batch, moe = c.p.experts > 0, nu = m ? job_size(m, &c.p) : -1;
	int trunk = trunk_round(&c.p, height + 1);
	size_t np = m ? tr_n_params(m) : 0;
	float *base = m ? malloc(np * sizeof(float)) : NULL;
	int *in = malloc((size_t)B * T * sizeof(int) + 1), *tg = malloc((size_t)B * T * sizeof(int) + 1);
	int *win = malloc((size_t)(T + 1) * sizeof(int) + 1);
	uint32_t *un = nu > 0 ? malloc((size_t)nu * sizeof *un) : NULL;
	if (!m || !base || !in || !tg || !win || !un || (nu = job_units(m, &c.p, height + 1, parent, key->pk, un)) < 1) goto out;
	if (too_short(d.n, c.p.shards, c.p.stripe, T)) { fprintf(stderr, "chain: the data is too short for this model\n"); goto out; }

	/* warm: the trainer's own optimizer, else the genesis one, else cold */
	char optp[PATHN], name[96];
	const char *from = "cold";
	uint64_t orng;
	snprintf(name, sizeof name, "trainers/%s.opt", trainer);
	if (pj(optp, dir, name)) goto out;
	if (access(optp, F_OK) == 0) { if (tr_load_opt(m, optp, &orng)) goto out; from = "its own optimizer"; }
	else {
		char gp[PATHN];
		if (pj(gp, dir, "genesis.opt")) goto out;
		if (access(gp, F_OK) == 0) { if (tr_load_opt(m, gp, &orng)) goto out; from = "the genesis optimizer"; }
	}

	uint32_t shard = ch_shard(parent, key->pk, c.p.shards);
	unsigned char jh[32];   /* the job's own randomness, as fixed as its shard */
	sha256_ctx sc; sha256_init(&sc);
	sha256_add(&sc, "spelaeum job\n", 13); sha256_add(&sc, parent, 32); sha256_add(&sc, key->pk, 32);
	sha256_done(&sc, jh);
	uint64_t rng = h64(jh);
	float *p = tr_params(m);
	memcpy(base, p, np * sizeof(float));
	int warm = c.p.steps / 10 > 0 ? c.p.steps / 10 : 1;
	double loss = 0;
	fprintf(stderr, "train: height %llu, shard %u of %d, %d steps of %d x %d, %s, on %s\n",
	        (unsigned long long)height, shard, c.p.shards, c.p.steps, B, T, from,
	        !tr_cfg(m)->experts && tr_gpu_name() ? tr_gpu_name() : "the CPU");
	if (trunk) fprintf(stderr, "  a trunk round: the trunk and its routers, every expert frozen\n");
	else if (moe) {   /* its experts and nothing else, on the loss the seal checks: with the routers frozen, aux
	                losses could only bend earlier experts to steer later routers */
		tr_set_aux(m, 0, 0);
		fprintf(stderr, "  experts");
		for (int k = 0; k < nu; k++) fprintf(stderr, " %u.%u", (un[k] - 1) / (uint32_t)tr_cfg(m)->experts, (un[k] - 1) % (uint32_t)tr_cfg(m)->experts);
		fprintf(stderr, " (layer.expert), the rest frozen\n");
	}
	if (moe) {   /* frozen for the optimizer too: its moments there wait, undecayed, for the round that trains them */
		size_t *live = malloc((size_t)nu * 2 * sizeof *live);
		if (!live) goto out;
		for (int k = 0; k < nu; k++) live[2 * k] = unit_at(m, moe, un[k], &live[2 * k + 1]);
		int bad = tr_set_live(m, live, nu);
		free(live);
		if (bad) goto out;
	}
	const char *ev = getenv("SPEL_EVERY");   /* steps between progress lines: 50, or 1 for a page showing each */
	int every = ev && atoi(ev) > 0 ? atoi(ev) : 50;
	/* sequences a pass: the whole batch, or fewer where memory is short (a tab on a phone);
	 * the passes add up to the same gradient, only summed in another order */
	const char *mv = getenv("SPEL_MICRO");
	int micro = mv && atoi(mv) > 0 && atoi(mv) < B ? atoi(mv) : B;
	if (micro < B) fprintf(stderr, "  in passes of %d\n", micro);
	for (int s = 0; s < c.p.steps; s++) {
		float lr = c.p.lr, lo_lr = lr / 10;
		float rate = s < warm ? lr * (s + 1) / warm
		           : lo_lr + 0.5f * (lr - lo_lr) * (1 + cosf(3.14159265f * (s - warm) / (float)(c.p.steps - warm > 0 ? c.p.steps - warm : 1)));
		tr_zero_grad(m);
		for (int b = 0; b < B; b++) {
			int64_t at = ch_window(d.n, c.p.shards, c.p.stripe, (int)shard, mix(rng++), T);
			if (ids_read(&d, at, T + 1, win)) { fprintf(stderr, "train: could not read the data\n"); goto out; }
			memcpy(in + (size_t)b * T, win, (size_t)T * sizeof(int));
			memcpy(tg + (size_t)b * T, win + 1, (size_t)T * sizeof(int));
		}
		double l = 0;
		for (int g = 0; g < B && l >= 0; g += micro) {
			int k = B - g < micro ? B - g : micro;
			double lg = tr_run(m, in + (size_t)g * T, tg + (size_t)g * T, k, T, 1.0f / ((float)B * T));
			l = lg < 0 ? -1 : l + lg;
		}
		if (l < 0 || !isfinite(l)) { fprintf(stderr, "train: the loss went bad at step %d\n", s); goto out; }
		loss = l / ((double)B * T);
		if (moe) outside(m, un, nu, tr_grads(m), NULL);   /* before the clip: frozen grads count for nothing */
		tr_clip(m, 1.0);
		tr_adamw(m, rate, 0.0f, 0.9f, 0.95f, 1e-8f);
		if ((s + 1) % every == 0 || s + 1 == c.p.steps) fprintf(stderr, "  step %d/%d  loss %.4f\n", s + 1, c.p.steps, loss);
	}
	p = tr_params(m);   /* the trained weights: they stay on the GPU while it trains */
	size_t nd = 0;   /* base is the delta now, packed unit by unit: the write never passes the read */
	for (int k = 0; k < nu; k++) {
		size_t len, off = unit_at(m, moe, un[k], &len);
		for (size_t j = 0; j < len; j++) base[nd++] = p[off + j] - base[off + j];
	}
	{	/* the optimizer goes home with the trainer, whatever the seal makes of the delta */
		char md[PATHN];
		if (pj(md, dir, "trainers") || (mkdir(md, 0755) != 0 && errno != EEXIST) || tr_save_opt(m, optp, rng)) goto out;
	}

	ch_dhead h;
	memset(&h, 0, sizeof h);
	memcpy(h.magic, "HKDL", 4); h.version = moe ? CH_UNITS : CH_VERSION;
	memcpy(h.parent, parent, 32);
	h.n = nd; h.shard = shard; h.steps = (uint32_t)c.p.steps; h.train_loss = loss;
	memcpy(h.trainer, trainer, strlen(trainer)); memcpy(h.key, key->pk, 32);
	char path[PATHN];
	snprintf(name, sizeof name, "pending/.train-%ld", (long)getpid());
	if (pj(path, dir, name)) goto out;
	uint32_t nun = (uint32_t)nu;
	unsigned char commit[32], sig[CH_SIG];
	part ps[5] = { { &h, sizeof h }, { &nun, moe ? 4 : 0 }, { un, moe ? (size_t)nu * 4 : 0 }, { base, nd * sizeof(float) }, { sig, sizeof sig } };
	sha256_ctx cc; sha256_init(&cc);   /* the commit is the file before its signature */
	for (int i = 0; i < 4; i++) sha256_add(&cc, ps[i].p, ps[i].n);
	sha256_done(&cc, commit);
	sign_commit(key, parent, commit, h.trainer, sig);
	if (write_hashed(path, ps, 5, NULL)) goto out;
	char final[PATHN];
	sha256_hex(commit, commit_hex);
	snprintf(name, sizeof name, "pending/%s.dlt", commit_hex);
	if (pj(final, dir, name) || rename(path, final) != 0) { perror(final); remove(path); goto out; }
	rc = 0;
out:
	free(base); free(in); free(tg); free(win); free(un);
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

/* the commit of a delta file: sha256 of what comes before its signature */
static unsigned char *slurp_commit(const char *path, size_t *len, unsigned char commit[32]) {
	unsigned char *b = slurp_hash(path, len, NULL);
	if (b && *len >= sizeof(ch_dhead) + CH_SIG) sha256(b, *len - CH_SIG, commit);
	else memset(commit, 0, 32);
	return b;
}

/* a pending delta checked against the tip: name = commit, signed by its key, shape,
 * shard, units, finite. un gets its units (job_size of them), the floats come back
 * packed in their order; sig, when not NULL, its signature */
static float *read_delta(const char *path, const unsigned char want[32], const unsigned char parent[32],
                         const tr_model *m, const ch_params *cp, uint64_t height, uint32_t *un, int *nun, ch_dhead *out,
                         unsigned char *sig, const char **why) {
	size_t len; unsigned char hash[32];
	unsigned char *b = slurp_commit(path, &len, hash);
	*why = "unreadable";
	if (!b) return NULL;
	ch_dhead *h = (ch_dhead *)b;
	int moe = cp->experts > 0, nu = 0;
	size_t at, n;
	if (len < sizeof *h + CH_SIG || memcmp(h->magic, "HKDL", 4)) { *why = "not a delta"; goto bad; }
	if (memchr(h->trainer, 0, CH_NAME) == NULL || !name_ok(h->trainer)) { *why = "bad trainer name"; goto bad; }
	if (memcmp(hash, want, 32)) { *why = "its name is not its hash"; goto bad; }
	if (memcmp(h->parent, parent, 32)) { *why = "stale (trained on another block)"; goto bad; }
	if (!sig_ok(h->key, parent, hash, h->trainer, b + len - CH_SIG)) { *why = "its signature is not its key's"; goto bad; }
	if ((nu = job_units(m, cp, height, parent, h->key, un)) < 1) { *why = "no units for it"; goto bad; }
	at = sizeof *h + (moe ? 4 + 4 * (size_t)nu : 0); n = units_len(m, moe, un, nu);
	if (h->version != (moe ? CH_UNITS : CH_VERSION)) { *why = moe ? "a dense delta on an expert chain" : "an expert delta on a dense chain"; goto bad; }
	if (h->n != n || len != at + n * 4 + CH_SIG) { *why = "wrong size for this model"; goto bad; }
	if (h->shard != ch_shard(parent, h->key, cp->shards)) { *why = "trained on a shard it was not given"; goto bad; }
	if (moe && (len < at || memcmp(b + sizeof *h, &(uint32_t){ (uint32_t)nu }, 4) || memcmp(b + sizeof *h + 4, un, 4 * (size_t)nu))) {
		*why = "it names units it was not given"; goto bad;
	}
	float *dl = (float *)(b + at);
	for (size_t j = 0; j < n; j++) if (!isfinite(dl[j])) { *why = "not finite"; goto bad; }
	*out = *h; *nun = nu;
	if (sig) memcpy(sig, b + len - CH_SIG, CH_SIG);
	memmove(b, dl, n * 4);   /* hand back the floats at the start of the buffer */
	return (float *)b;
bad:
	free(b);
	return NULL;
}

/* the round's deltas: every pending one checked against the tip, the bad and the
 * stale set aside, a key's second in a round set aside too; sorted by commit, into
 * *out (malloc'd). un: room for a job's units. The count, or -1 */
static int gather(const char *dir, const ch_params *cp, const tr_model *m, uint64_t height, const unsigned char parent[32],
                  uint32_t *un, pend **out) {
	char path[PATHN];
	pend *ps = NULL;
	int np_ = 0, cap = 0, nu = 0;
	*out = NULL;
	DIR *dd = pj(path, dir, "pending") ? NULL : opendir(path);
	if (!dd) { perror(path); return -1; }
	struct dirent *e;
	while ((e = readdir(dd))) {
		size_t nl = strlen(e->d_name);
		if (nl != 68 || strcmp(e->d_name + 64, ".dlt")) continue;
		if (np_ == cap) {
			cap = cap ? cap * 2 : 8;
			pend *nb = realloc(ps, (size_t)cap * sizeof *ps);
			if (!nb) { free(ps); closedir(dd); return -1; }
			ps = nb;
		}
		pend *q = &ps[np_];
		memset(q, 0, sizeof *q);
		char name[96];
		snprintf(name, sizeof name, "pending/%s", e->d_name);
		if (pj(q->path, dir, name) || unhex(e->d_name, q->c.commit)) continue;
		ch_dhead h; const char *why;
		float *dl = read_delta(q->path, q->c.commit, parent, m, cp, height + 1, un, &nu, &h, q->c.sig, &why);
		if (!dl) { set_aside(q->path, why, strstr(why, "stale") ? "stale" : "bad"); continue; }
		free(dl);
		memcpy(q->c.trainer, h.trainer, CH_NAME); memcpy(q->c.key, h.key, 32); q->c.shard = h.shard;
		np_++;
	}
	closedir(dd);
	if (np_) qsort(ps, (size_t)np_, sizeof *ps, cmp_pend);
	int k = 0;
	for (int i = 0; i < np_; i++) {
		int dup = 0;
		for (int j = 0; j < k; j++) dup |= !memcmp(ps[j].c.key, ps[i].c.key, 32);
		if (dup) set_aside(ps[i].path, "a second delta from the same key this round", "dup");
		else ps[k++] = ps[i];
	}
	*out = ps;
	return k;
}

/* ── checks: trainers check each other's deltas, each check a signed verdict ── */

/* what a checker signs: the parent, the delta's commit, and the loss it found */
#define VSIGNED (15 + 32 + 32 + 8)
static void verdict_msg(const unsigned char parent[32], const ch_verdict *v, unsigned char out[VSIGNED]) {
	memcpy(out, "spelaeum check\n", 15); memcpy(out + 15, parent, 32); memcpy(out + 47, v->commit, 32); memcpy(out + 79, &v->loss, 8);
}
void ch_verdict_sign(const ch_key *k, const unsigned char parent[32], ch_verdict *v) {
	unsigned char msg[VSIGNED];
	memcpy(v->key, k->pk, 32);
	verdict_msg(parent, v, msg);
	ed25519_sign(v->sig, msg, sizeof msg, k->seed, k->pk);
}
int ch_verdict_ok(const unsigned char parent[32], const ch_verdict *v) {
	unsigned char msg[VSIGNED];
	verdict_msg(parent, v, msg);
	return isfinite(v->loss) && v->loss >= 0 && ed25519_verify(v->sig, msg, sizeof msg, v->key) == 0;
}

/* keys, one in hex a line, from DIR/name: the checkers (keys with a passed job) or the barred */
static int keys_load(const char *dir, const char *name, unsigned char (**out)[32]) {
	char path[PATHN], line[160];
	unsigned char (*ks)[32] = NULL;
	int n = 0, cap = 0;
	*out = NULL;
	FILE *f = pj(path, dir, name) ? NULL : fopen(path, "r");
	while (f && fgets(line, sizeof line, f)) {
		unsigned char k[32];
		if (strlen(line) < 64 || unhex(line, k)) continue;
		if (n == cap) {
			cap = cap ? cap * 2 : 64;
			unsigned char (*nk)[32] = realloc(ks, (size_t)cap * 32);
			if (!nk) { free(ks); fclose(f); return -1; }
			ks = nk;
		}
		memcpy(ks[n++], k, 32);
	}
	if (f) fclose(f);
	*out = ks;
	return n;
}
static int key_in(unsigned char (*ks)[32], int n, const unsigned char k[32]) {
	for (int i = 0; i < n; i++) if (!memcmp(ks[i], k, 32)) return 1;
	return 0;
}
static void keys_add(const char *dir, const char *name, const unsigned char k[32]) {
	char path[PATHN], hx[65];
	FILE *f = pj(path, dir, name) ? NULL : fopen(path, "a");
	if (!f) return;
	sha256_hex(k, hx);   /* hex, nothing hashed */
	fprintf(f, "%s\n", hx);
	fclose(f);
}
int ch_checker(const char *dir, const unsigned char key[32]) {
	unsigned char (*ck)[32], (*br)[32];
	int nc = keys_load(dir, "checkers", &ck), nb = keys_load(dir, "barred", &br);
	int ok = nc > 0 && key_in(ck, nc, key) && !(nb > 0 && key_in(br, nb, key));
	free(ck); free(br);
	return ok;
}

/* the losses k checkers agree on for one delta: verdicts from pending/<commit>.vrd
 * signed by a checker, not barred, not the delta's own key, one a key. Their median
 * into *loss and their keys into agree (room for NET-sized rounds: at most cap);
 * returns how many agree with the median */
static int agreed(const char *dir, const pend *q, const unsigned char parent[32], unsigned char (*ck)[32], int nc,
                  unsigned char (*br)[32], int nb, double *loss, unsigned char (*agree)[32], int cap) {
	char path[PATHN], name[96], hx[65];
	sha256_hex(q->c.commit, hx);   /* hex, nothing hashed */
	snprintf(name, sizeof name, "pending/%s.vrd", hx);
	FILE *f = pj(path, dir, name) ? NULL : fopen(path, "rb");
	if (!f) return 0;
	ch_verdict v, vs[64];
	int n = 0;
	while (n < 64 && fread(&v, sizeof v, 1, f) == 1) {
		if (memcmp(v.commit, q->c.commit, 32) || !memcmp(v.key, q->c.key, 32) || !key_in(ck, nc, v.key) || key_in(br, nb, v.key)) continue;
		int seen = 0;
		for (int i = 0; i < n; i++) seen |= !memcmp(vs[i].key, v.key, 32);
		if (!seen && ch_verdict_ok(parent, &v)) vs[n++] = v;
	}
	fclose(f);
	if (!n) return 0;
	double ls[64];
	for (int i = 0; i < n; i++) ls[i] = vs[i].loss;
	for (int i = 1; i < n; i++) for (int j = i; j > 0 && ls[j - 1] > ls[j]; j--) { double t = ls[j]; ls[j] = ls[j - 1]; ls[j - 1] = t; }
	double med = ls[n / 2];
	int a = 0;
	for (int i = 0; i < n; i++) if (vs[i].loss == med) { if (a < cap) memcpy(agree[a], vs[i].key, 32); a++; }
	*loss = med;
	return a;
}

static int seal(const char *dir, int need, int spot, int *taken);
int ch_seal(const char *dir) { return seal(dir, 0, 0, NULL); }
int ch_seal_checked(const char *dir, int need, int spot, int *taken) { return seal(dir, need, spot, taken); }

static int seal(const char *dir, int need, int spot, int *taken) {
	ch_config c; ch_ids d = {0};
	uint64_t height; unsigned char parent[32];
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, parent) || ids_open(dir, &d)) { ids_close(&d); return -1; }
	double diff;
	if (next_difficulty(dir, &c, height, &diff)) { ids_close(&d); return -1; }
	void *blob = NULL; size_t blen = 0; int eos = 0;
	tr_model *m = load_tip(dir, &blob, &blen, &eos);
	int rc = -1, T = m ? tr_cfg(m)->ctx : 0, moe = c.p.experts > 0, nu = m ? job_size(m, &c.p) : -1;
	int nall = moe && m ? 1 + tr_cfg(m)->layers * tr_cfg(m)->experts : 1;   /* units in the model */
	unsigned char (*ck)[32] = NULL, (*br)[32] = NULL, agree[64][32];
	int nc = 0, nbar = 0;
	if (taken) *taken = 0;
	size_t np = m ? tr_n_params(m) : 0;
	pend *ps = NULL;
	float *base = NULL, *sum = NULL, *use = NULL, *keep = NULL;
	int64_t *at = NULL;
	uint32_t *skip = NULL, *un = NULL, *ub = NULL, *um = NULL, *bu = NULL;   /* a delta's, the best's, the merge's, the block's */
	unsigned char (*commits)[32] = NULL;
	int *cnt = NULL, nb = 0;
	DIR *dd = NULL;
	char path[PATHN];
	if (!m || nu < 0) goto out;
	if (need > 0 && ((nc = keys_load(dir, "checkers", &ck)) < 0 || (nbar = keys_load(dir, "barred", &br)) < 0)) goto out;
	un = malloc((size_t)nu * sizeof *un); ub = malloc((size_t)nu * sizeof *ub);
	um = malloc((size_t)nall * sizeof *um); cnt = calloc((size_t)nall, sizeof *cnt);
	if (!un || !ub || !um || !cnt) goto out;

	int k = gather(dir, &c.p, m, height, parent, un, &ps);
	if (k < 0) goto out;
	if (!k) { fprintf(stderr, "seal: nothing pending above height %llu\n", (unsigned long long)height); rc = 1; goto out; }

	/* the seed from every commit, then the windows it picks */
	commits = malloc((size_t)k * 32); skip = malloc((size_t)k * sizeof *skip);
	at = malloc((size_t)c.p.eval_windows * sizeof *at);
	base = malloc(np * sizeof(float)); sum = calloc(np, sizeof(float));
	if (!commits || !skip || !at || !base || !sum) goto out;
	for (int i = 0; i < k; i++) { memcpy(commits[i], ps[i].c.commit, 32); skip[i] = ps[i].c.shard; }
	unsigned char seed[32];
	ch_seed(parent, (const unsigned char (*)[32])commits, k, seed);
	if (pick_windows(seed, &d, c.p.shards, c.p.stripe, skip, k, T, c.p.eval_windows, at)) goto out;

	float *p = tr_params(m);
	memcpy(base, p, np * sizeof(float));
	double pl = eval_loss(m, &d, at, c.p.eval_windows, T);
	if (pl < 0) goto out;
	fprintf(stderr, "seal: height %llu -> %llu, %d deltas, difficulty %.5f, parent loss %.4f on %d windows\n",
	        (unsigned long long)height, (unsigned long long)height + 1, k, diff, pl, c.p.eval_windows);
	int passed = 0, best = -1;
	for (int i = 0; i < k; i++) {
		ch_dhead h; const char *why;
		float *dl = read_delta(ps[i].path, ps[i].c.commit, parent, m, &c.p, height + 1, un, &nu, &h, NULL, &why);
		if (!dl) goto out;
		/* k checkers that agree stand for the work, but for one in every `spot` (picked by
		 * the seed, so nobody knows which before the round closes), worked out anyway */
		double vl = 0, l = -1;
		int na = need > 0 ? agreed(dir, &ps[i], parent, ck, nc, br, nbar, &vl, agree, 64) : 0, from = 0;
		unsigned char sp[32];
		sha256_ctx sc; sha256_init(&sc); sha256_add(&sc, seed, 32); sha256_add(&sc, ps[i].c.commit, 32); sha256_done(&sc, sp);
		int spotted = spot > 0 && mix(h64(sp)) % (uint64_t)spot == 0;
		if (na >= need && need > 0 && !spotted) { l = vl; from = 1; }
		else {
			apply(m, moe, un, nu, p, base, dl);
			l = eval_loss(m, &d, at, c.p.eval_windows, T);
			apply(m, moe, un, nu, p, base, NULL);
			if (l < 0) { free(dl); goto out; }
			if (na >= need && need > 0 && vl != l) {   /* they agreed on a loss that is not the delta's */
				for (int a = 0; a < na && a < 64; a++) { keys_add(dir, "barred", agree[a]); if (nbar >= 0) { unsigned char (*nb2)[32] = realloc(br, (size_t)(nbar + 1) * 32); if (nb2) { br = nb2; memcpy(br[nbar++], agree[a], 32); } } }
				fprintf(stderr, "  %-20s %d checkers said %.4f, the delta's loss is %.4f: barred from checking\n", ps[i].c.trainer, na, vl, l);
			}
		}
		if (from && taken) (*taken)++;
		ps[i].c.loss = l;
		ps[i].c.state = pl - l >= diff ? CH_PASSED : CH_REJECTED;
		if (ps[i].c.state == CH_PASSED) {
			apply(m, moe, un, nu, sum, NULL, dl);
			for (int q = 0; q < nu; q++) cnt[un[q]]++;
			passed++;
			if (best < 0 || l < ps[best].c.loss) best = i;
		}
		fprintf(stderr, "  %-20s shard %3u  loss %.4f  cut %+.4f  %s%s\n", ps[i].c.trainer, ps[i].c.shard, l, pl - l,
		        ps[i].c.state == CH_PASSED ? "passes" : "short of the difficulty", from ? " (checked by trainers)" : "");
		free(dl);
	}

	/* the block's delta: per unit, the mean of the passes that carry it (one carrier: its
	 * floats as they are), when that beats the best one alone */
	double nl = pl;
	size_t nd = 0;
	if (passed) {
		ch_dhead h; const char *why;
		keep = read_delta(ps[best].path, ps[best].c.commit, parent, m, &c.p, height + 1, ub, &nb, &h, NULL, &why);
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
			double ml = eval_loss(m, &d, at, c.p.eval_windows, T);
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
	sha256(use ? (const void *)use : "", nd * sizeof(float), bh.body);
	model_hash(m, bh.model);
	unsigned char bhash[32];   /* the head part's: the floats are in it by their hash */
	sha256_ctx hc; sha256_init(&hc);
	for (int i = 0; i < 4; i++) if (bp[i].n) sha256_add(&hc, bp[i].p, bp[i].n);
	sha256_done(&hc, bhash);
	int wr = block_path(dir, bh.height, path) || write_hashed(path, bp, 5, NULL);
	free(cs);
	if (wr) goto out;
	if (pj(path, dir, "tip.mlf2") || tr_save(m, path, blob, blen, eos) || write_tip(dir, bh.height, bhash)) goto out;
	{	/* keys with a passed job may check from now on; this round's verdicts and its round file go */
		unsigned char (*all)[32] = NULL;
		int na = keys_load(dir, "checkers", &all);
		for (int i = 0; i < k; i++) {
			if (ps[i].c.state != CH_REJECTED && na >= 0 && !key_in(all, na, ps[i].c.key)) {
				keys_add(dir, "checkers", ps[i].c.key);
				unsigned char (*nk)[32] = realloc(all, (size_t)(na + 1) * 32);
				if (nk) { all = nk; memcpy(all[na++], ps[i].c.key, 32); }
			}
			char vp[PATHN + 8];
			snprintf(vp, sizeof vp, "%.*s.vrd", (int)(strlen(ps[i].path) - 4), ps[i].path);
			remove(ps[i].path); remove(vp);
		}
		free(all);
		if (pj(path, dir, "round") == 0) remove(path);
	}
	char hx[65];
	sha256_hex(bhash, hx);
	fprintf(stderr, "seal: block %llu %.16s, %u of %d merged, loss %.4f -> %.4f, next difficulty %.5f%s\n",
	        (unsigned long long)bh.height, hx, bh.nmerged, k, pl, nl, ch_retarget(diff, k, passed),
	        c.p.trunk > 0 ? trunk_round(&c.p, bh.height) ? " for trunk rounds" : " for expert rounds" : "");
	rc = 0;
out:
	if (dd) closedir(dd);
	free(ps); free(base); free(sum); free(keep); free(at); free(skip); free(commits); free(blob);
	free(un); free(ub); free(um); free(cnt); free(ck); free(br);
	tr_free(m); ids_close(&d);
	return rc;
}

int ch_close(const char *dir) {
	ch_config c;
	uint64_t height; unsigned char parent[32];
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, parent)) return -1;
	tr_model *m = load_tip(dir, NULL, NULL, NULL);
	int nu = m ? job_size(m, &c.p) : -1, rc = -1;
	uint32_t *un = nu > 0 ? malloc((size_t)nu * sizeof *un) : NULL;
	pend *ps = NULL;
	unsigned char (*commits)[32] = NULL;
	char path[PATHN], tmp[PATHN + 8], hx[65], kx[65];
	int k = un ? gather(dir, &c.p, m, height, parent, un, &ps) : -1;
	if (k < 0) goto out;
	if (k == 0) { rc = 1; goto out; }
	if (!(commits = malloc((size_t)k * 32))) goto out;
	for (int i = 0; i < k; i++) memcpy(commits[i], ps[i].c.commit, 32);
	unsigned char seed[32];
	ch_seed(parent, (const unsigned char (*)[32])commits, k, seed);
	if (pj(path, dir, "round")) goto out;
	snprintf(tmp, sizeof tmp, "%s.part", path);
	FILE *f = fopen(tmp, "w");
	if (!f) goto out;
	sha256_hex(parent, hx);
	fprintf(f, "spelaeum round\nheight %llu\nparent %s\n", (unsigned long long)height, hx);
	sha256_hex(seed, hx);
	fprintf(f, "seed %s\n", hx);
	for (int i = 0; i < k; i++) {
		sha256_hex(ps[i].c.commit, hx); sha256_hex(ps[i].c.key, kx);   /* hex, nothing hashed */
		fprintf(f, "delta %s %s %u\n", hx, kx, ps[i].c.shard);
	}
	if (fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); goto out; }
	rc = 0;
out:
	free(ps); free(commits); free(un); tr_free(m);
	return rc;
}

/* a round file: its parent, seed, and each delta's commit, key and shard (malloc'd) */
static int read_round(const char *path, unsigned char parent[32], unsigned char seed[32], unsigned char (**cm)[32],
                      unsigned char (**ks)[32], uint32_t **sh) {
	char line[300], a[80], b[80];
	unsigned sv;
	int n = 0, cap = 0, ok = 0;
	*cm = NULL; *ks = NULL; *sh = NULL;
	FILE *f = fopen(path, "r");
	if (!f) return -1;
	if (fgets(line, sizeof line, f) && !strcmp(line, "spelaeum round\n")) ok = 1;
	while (ok && fgets(line, sizeof line, f)) {
		if (sscanf(line, "parent %79s", a) == 1) ok = strlen(a) == 64 && !unhex(a, parent);
		else if (sscanf(line, "seed %79s", a) == 1) ok = strlen(a) == 64 && !unhex(a, seed);
		else if (sscanf(line, "delta %79s %79s %u", a, b, &sv) == 3) {
			if (n == cap) {
				cap = cap ? cap * 2 : 16;
				void *x = realloc(*cm, (size_t)cap * 32), *y = x ? realloc(*ks, (size_t)cap * 32) : NULL, *z = y ? realloc(*sh, (size_t)cap * 4) : NULL;
				if (x) *cm = x;
				if (y) *ks = y;
				if (z) *sh = z;
				if (!x || !y || !z) { ok = 0; break; }
			}
			ok = strlen(a) == 64 && strlen(b) == 64 && !unhex(a, (*cm)[n]) && !unhex(b, (*ks)[n]);
			(*sh)[n++] = sv;
		}
	}
	fclose(f);
	if (!ok || !n) { free(*cm); free(*ks); free(*sh); *cm = NULL; *ks = NULL; *sh = NULL; return -1; }
	return n;
}

int ch_check(const char *dir, const char *round, const char *delta, unsigned char parent_out[32], ch_verdict *v) {
	ch_config c; ch_ids d = {0};
	uint64_t height; unsigned char parent[32], rp[32], seed[32], want[32], commit[32];
	unsigned char (*cm)[32] = NULL, (*ks)[32] = NULL;
	uint32_t *sh = NULL, *un = NULL;
	tr_model *m = NULL;
	int64_t *at = NULL;
	int rc = -1;
	float *dl = NULL;
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, parent)) return -1;
	int n = read_round(round, rp, seed, &cm, &ks, &sh);
	if (n < 1) { fprintf(stderr, "check: %s is not a round\n", round); return -1; }
	if (memcmp(rp, parent, 32)) { fprintf(stderr, "check: this copy is not at the round's parent\n"); goto out; }
	/* the seed and the shards are worked out here, not taken on trust */
	ch_seed(parent, (const unsigned char (*)[32])cm, n, want);
	int good = !memcmp(want, seed, 32);
	for (int i = 0; good && i < n; i++) good = sh[i] == ch_shard(parent, ks[i], c.p.shards) && (i == 0 || memcmp(cm[i - 1], cm[i], 32) < 0);
	if (!good) { fprintf(stderr, "check: the round's seed or shards are not its deltas'\n"); goto out; }
	size_t len;
	unsigned char *b = slurp_commit(delta, &len, commit);
	free(b);
	int at_ = -1;
	for (int i = 0; i < n; i++) if (!memcmp(cm[i], commit, 32)) at_ = i;
	if (at_ < 0) { fprintf(stderr, "check: that delta is not in the round\n"); goto out; }
	if (!(m = load_tip(dir, NULL, NULL, NULL)) || ids_open(dir, &d)) goto out;
	int T = tr_cfg(m)->ctx, nu = job_size(m, &c.p), moe = c.p.experts > 0;
	if (nu < 1 || !(un = malloc((size_t)nu * sizeof *un))) goto out;
	ch_dhead h; const char *why;
	if (!(dl = read_delta(delta, commit, parent, m, &c.p, height + 1, un, &nu, &h, NULL, &why))) { fprintf(stderr, "check: the delta: %s\n", why); goto out; }
	at = malloc((size_t)c.p.eval_windows * sizeof *at);
	if (!at || pick_windows(seed, &d, c.p.shards, c.p.stripe, sh, n, T, c.p.eval_windows, at)) goto out;
	apply(m, moe, un, nu, tr_params(m), NULL, dl);
	double l = eval_loss(m, &d, at, c.p.eval_windows, T);
	if (l < 0) goto out;
	memset(v, 0, sizeof *v);
	memcpy(v->commit, commit, 32);
	v->loss = l;
	if (parent_out) memcpy(parent_out, parent, 32);
	rc = 0;
out:
	free(cm); free(ks); free(sh); free(un); free(dl); free(at);
	tr_free(m); ids_close(&d);
	return rc;
}

/* ── verify ─────────────────────────────────────────────────────────────── */

/* contributions are in commit order, so a key's second delta can sit anywhere before */
static int key_before(const ch_contrib *cs, uint32_t i) {
	for (uint32_t j = 0; j < i; j++) if (!memcmp(cs[j].key, cs[i].key, 32)) return 1;
	return 0;
}

/* what a block's head part shows on its own, against the block before it (prev, at
 * h - 1): height, link, difficulty, every contribution (order, name, one a key,
 * shard, signature, state against its loss), the merged count, the units (the union
 * of its merged trainers', in order: none past the end, the trunk in a trunk round
 * only) and the seed. skip gets each contribution's shard. NULL when it holds. */
static const char *check_head(const char *dir, const ch_config *c, const tr_model *m, uint64_t h, const unsigned char prev[32],
                              const ch_bhead *hd, const ch_contrib *cs, const uint32_t *bu, uint32_t nb, uint32_t *skip) {
	int moe = c->p.experts > 0, nu = job_size(m, &c->p), nall = moe ? 1 + tr_cfg(m)->layers * tr_cfg(m)->experts : 1;
	unsigned char (*cm)[32] = malloc((size_t)hd->ncontrib * 32 + 1), *mark = malloc((size_t)nall), seed[32];
	uint32_t *un = nu > 0 ? malloc((size_t)nu * sizeof *un) : NULL;
	const char *bad = NULL;
	int merged = 0;
	double want = 0;
	if (!cm || !mark || !un) bad = "out of memory";
	else if (hd->height != h) bad = "wrong height";
	else if (memcmp(hd->parent, prev, 32)) bad = "its parent is not the block before it";
	else if (next_difficulty(dir, c, h - 1, &want)) bad = "the block its difficulty comes from is unreadable";
	else if (hd->difficulty != want) bad = "wrong difficulty";
	else if (!moe && hd->n != 0 && hd->n != tr_n_params(m)) bad = "wrong delta size";
	for (uint32_t i = 0; !bad && i < hd->ncontrib; i++) {
		const ch_contrib *x = &cs[i];
		if (i && memcmp(cs[i - 1].commit, x->commit, 32) >= 0) bad = "deltas out of commit order";
		else if (!memchr(x->trainer, 0, CH_NAME) || !name_ok(x->trainer)) bad = "bad trainer name";
		else if (key_before(cs, i)) bad = "two deltas from one key";
		else if (x->shard != ch_shard(prev, x->key, c->p.shards)) bad = "a delta on a shard it was not given";
		else if (!sig_ok(x->key, prev, x->commit, x->trainer, x->sig)) bad = "a delta its key did not sign";
		else if (x->state > CH_MERGED || (x->state != CH_REJECTED) != (hd->parent_loss - x->loss >= hd->difficulty))
			bad = "a delta's state does not match its loss";
		memcpy(cm[i], x->commit, 32); skip[i] = x->shard;
		merged += x->state == CH_MERGED;
	}
	if (!bad && (merged != (int)hd->nmerged || (merged == 0) != (hd->n == 0))) bad = "merged count does not match the delta";
	if (!bad && moe) {
		memset(mark, 0, (size_t)nall);
		for (uint32_t i = 0; i < hd->ncontrib; i++)
			if (cs[i].state == CH_MERGED && (nu = job_units(m, &c->p, h, prev, cs[i].key, un)) > 0)
				for (int q = 0; q < nu; q++) mark[un[q]] = 1;
		uint32_t q = 0;
		int same = 1;
		for (int u = 0; u < nall; u++) if (mark[u]) same &= q < nb && bu[q++] == (uint32_t)u;
		if (!same || q != nb) bad = "its units are not the ones its merged trainers were given";
		else if (hd->n != units_len(m, moe, bu, (int)nb)) bad = "wrong delta size";
	}
	if (!bad) {
		ch_seed(prev, (const unsigned char (*)[32])cm, (int)hd->ncontrib, seed);
		if (memcmp(seed, hd->seed, 32)) bad = "wrong seed";
	}
	free(cm); free(mark); free(un);
	return bad;
}

/* the model hash a copy at height h must have: the head's, or genesis's own */
static int model_at(const char *dir, const ch_config *c, uint64_t h, unsigned char out[32]) {
	char path[PATHN];
	if (h == 0) {
		tr_model *g = pj(path, dir, "genesis.mlf2") ? NULL : tr_load(path, NULL, NULL, NULL);
		if (!g) return -1;
		model_hash(g, out);
		tr_free(g);
		return 0;
	}
	unsigned char *b; ch_bhead *hd; ch_contrib *cs; const uint32_t *bu; uint32_t nb; unsigned char hash[32];
	if (block_path(dir, h, path) || read_block(path, c->p.experts > 0, &b, &hd, &cs, &bu, &nb, NULL, hash)) return -1;
	memcpy(out, hd->model, 32);
	free(b);
	return 0;
}

/* data and genesis: the text a trainer trains on is the chain's, and so is block 0 */
static int check_genesis(const char *dir, const ch_config *c, const ch_ids *d) {
	unsigned char dh[32], gh[32];
	if (data_hash(&d->h, NULL, d, c->p.stripe, dh)) return -1;
	if (memcmp(dh, c->data, 32)) { fprintf(stderr, "verify: ids.bin is not the data this chain was made on\n"); return -1; }
	if (genesis_hash(dir, &c->p, dh, gh) || memcmp(gh, c->genesis, 32)) { fprintf(stderr, "verify: genesis.mlf2 or chain.cfg changed\n"); return -1; }
	return 0;
}

enum { VERIFY, ADVANCE, JOIN };
/* Walk the blocks above a start, each head checked. VERIFY: from genesis, every delta
 * applied (full: every loss re-run and every block's model hash), then tip and
 * tip.mlf2 must be what the walk made. ADVANCE: from this copy's tip, the new blocks
 * applied the same way, then tip and tip.mlf2 written. JOIN: from this copy's tip,
 * or genesis when it has none, heads only; then `model`, fetched whole, becomes
 * tip.mlf2 when its hash is the last head's. */
static int walk(const char *dir, int mode, int full, const char *model) {
	ch_config c; ch_ids d = {0};
	uint64_t h = 0, tip_h = 0; unsigned char prev[32], tip_hash[32], want[32];
	void *blob = NULL; size_t blen = 0; int eos = 0;
	char path[PATHN], mp[PATHN];
	int rc = -1, from_tip = mode == ADVANCE || (mode == JOIN && pj(path, dir, "tip") == 0 && access(path, F_OK) == 0);
	tr_model *m = NULL, *t = NULL;
	int64_t *at = NULL;
	if (ch_load_config(dir, &c)) return -1;
	if (mode == VERIFY && ch_tip(dir, &tip_h, tip_hash)) return -1;
	if (from_tip ? ch_tip(dir, &h, prev) : 0) return -1;
	if (!from_tip || full) {   /* from genesis, or re-running losses: the data is needed, and checked */
		if (ids_open(dir, &d) || check_genesis(dir, &c, &d)) goto out;
		if (!from_tip) memcpy(prev, c.genesis, 32);
	}
	if (mode == JOIN) { if (pj(mp, dir, model) || !(m = tr_load(mp, NULL, NULL, NULL))) goto out; }
	else if (mode == ADVANCE) {   /* the copy's own model must be the one its tip names */
		if (!(m = load_tip(dir, &blob, &blen, &eos)) || model_at(dir, &c, h, want)) goto out;
		unsigned char mh[32];
		model_hash(m, mh);
		if (memcmp(mh, want, 32)) { fprintf(stderr, "chain: tip.mlf2 is not the model block %llu names\n", (unsigned long long)h); goto out; }
	} else if (pj(path, dir, "genesis.mlf2") || !(m = tr_load(path, NULL, NULL, NULL))) goto out;
	int T = tr_cfg(m)->ctx, moe = c.p.experts > 0;
	float *p = tr_params(m);
	if (job_size(m, &c.p) < 0) goto out;
	at = malloc((size_t)c.p.eval_windows * sizeof *at);
	if (!at) goto out;
	for (;;) {
		if (block_path(dir, h + 1, path)) goto out;
		if (access(path, F_OK) != 0) break;
		h++;
		unsigned char *b; ch_bhead *hd; ch_contrib *cs; const float *dl = NULL; const uint32_t *bu; uint32_t nb; unsigned char bhash[32], mh[32];
		if (read_block(path, moe, &b, &hd, &cs, &bu, &nb, mode == JOIN ? NULL : &dl, bhash)) goto out;
		static const uint32_t whole = 0;   /* a dense block's one unit, when it carries a delta */
		if (!moe) { bu = &whole; nb = hd->n ? 1 : 0; }
		uint32_t *skip = malloc((size_t)hd->ncontrib * sizeof *skip + 1);
		const char *bad = skip ? check_head(dir, &c, m, h, prev, hd, cs, bu, nb, skip) : "out of memory";
		if (!bad && mode != JOIN && !dl) bad = "its delta was pruned from this copy: a replay needs every block whole";
		if (!bad && full) {
			if (pick_windows(hd->seed, &d, c.p.shards, c.p.stripe, skip, (int)hd->ncontrib, T, c.p.eval_windows, at)) bad = "no windows";
			else {
				double pl = eval_loss(m, &d, at, c.p.eval_windows, T);
				if (pl != hd->parent_loss) bad = "parent loss does not reproduce";
			}
		}
		if (!bad && mode != JOIN) apply(m, moe, bu, (int)nb, p, NULL, dl);
		if (!bad && full && hd->n) {
			double nl = eval_loss(m, &d, at, c.p.eval_windows, T);
			if (nl != hd->new_loss) bad = "the block's loss does not reproduce";
		}
		if (!bad && full && (model_hash(m, mh), memcmp(mh, hd->model, 32))) bad = "its model is not the one its head names";
		free(skip); free(b);
		if (bad) { fprintf(stderr, "verify: block %llu: %s\n", (unsigned long long)h, bad); goto out; }
		memcpy(prev, bhash, 32);
	}
	/* the model the walk ends on must be the one the last head names (genesis's at 0) */
	unsigned char mh[32];
	model_hash(m, mh);
	if (model_at(dir, &c, h, want) || memcmp(mh, want, 32)) {
		fprintf(stderr, mode == JOIN ? "chain: the fetched model is not the one block %llu names\n" : "verify: the model at block %llu is not the one its head names\n",
		        (unsigned long long)h);
		goto out;
	}
	if (mode == ADVANCE) {
		if (pj(path, dir, "tip.mlf2") || tr_save(m, path, blob, blen, eos) || write_tip(dir, h, prev)) goto out;
		rc = 0; goto out;
	}
	if (mode == JOIN) {
		if (pj(path, dir, "tip.mlf2") || rename(mp, path) != 0 || write_tip(dir, h, prev)) goto out;
		rc = 0; goto out;
	}
	if (h != tip_h || memcmp(prev, tip_hash, 32)) { fprintf(stderr, "verify: tip says %llu, the blocks end at %llu or hash elsewhere\n", (unsigned long long)tip_h, (unsigned long long)h); goto out; }
	if (pj(path, dir, "tip.mlf2") || !(t = tr_load(path, NULL, NULL, NULL)) || tr_n_params(t) != tr_n_params(m) ||
	    memcmp(tr_params(t), p, tr_n_params(m) * sizeof(float))) { fprintf(stderr, "verify: tip.mlf2 is not the replayed model\n"); goto out; }
	char hx[65];
	sha256_hex(prev, hx);
	fprintf(stderr, "verify: the chain holds, %llu blocks, tip %.16s%s\n", (unsigned long long)h, hx, full ? ", every loss reproduced" : "");
	rc = 0;
out:
	free(at); free(blob);
	tr_free(m); tr_free(t); ids_close(&d);
	return rc;
}
int ch_verify(const char *dir, int full) { return walk(dir, VERIFY, full, NULL); }
int ch_advance(const char *dir) { return walk(dir, ADVANCE, 0, NULL); }
int ch_join(const char *dir, const char *model) { return walk(dir, JOIN, 0, model); }

int ch_head_part(const char *dir, uint64_t h, unsigned char **out, size_t *len) {
	ch_config c;
	char path[PATHN];
	unsigned char *b; ch_bhead *hd; ch_contrib *cs; const uint32_t *bu; uint32_t nb; unsigned char hash[32];
	if (ch_load_config(dir, &c) || block_path(dir, h, path) ||
	    read_block_at(path, c.p.experts > 0, &b, &hd, &cs, &bu, &nb, NULL, hash, len)) return -1;
	*out = b;
	return 0;
}

int ch_put_heads(const char *dir, const unsigned char *buf, size_t len, uint64_t from, uint64_t upto, uint64_t *got) {
	ch_config c;
	char path[PATHN];
	*got = 0;
	if (ch_load_config(dir, &c)) return -1;
	int moe = c.p.experts > 0;
	for (size_t at = 0; at < len; (*got)++) {
		ch_bhead hd;
		uint32_t n = 0;
		if (len - at < sizeof hd) return -1;
		memcpy(&hd, buf + at, sizeof hd);
		if (memcmp(hd.magic, "HKBK", 4) || hd.height != from + *got || hd.ncontrib > 1u << 20) return -1;
		size_t ca = at + sizeof hd + (size_t)hd.ncontrib * sizeof(ch_contrib);
		if (moe) { if (ca > len || len - ca < 4) return -1; memcpy(&n, buf + ca, 4); if (n > (len - ca - 4) / 4) return -1; }
		size_t hl = head_len(&hd, moe, n);
		if (hl > len - at) return -1;
		if (hd.height > upto) break;   /* the rest are for later: a copy joining at a mirror's model */
		if (block_path(dir, hd.height, path)) return -1;
		if (access(path, F_OK) != 0) { part pt = { buf + at, hl }; if (write_hashed(path, &pt, 1, NULL)) return -1; }
		at += hl;
	}
	return 0;
}

int ch_prune(const char *dir, uint64_t keep, int quiet) {
	ch_config c;
	uint64_t height; unsigned char tip[32];
	char path[PATHN];
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, tip)) return -1;
	unsigned long long freed = 0, n = 0;
	for (uint64_t h = 1; h + keep <= height; h++) {
		unsigned char *b; ch_bhead *hd; ch_contrib *cs; const uint32_t *bu; uint32_t nb; unsigned char hash[32];
		size_t hl;
		struct stat st;
		if (block_path(dir, h, path) || stat(path, &st) != 0 ||
		    read_block_at(path, c.p.experts > 0, &b, &hd, &cs, &bu, &nb, NULL, hash, &hl)) return -1;
		free(b);
		if ((size_t)st.st_size == hl) continue;
		if (truncate(path, (off_t)hl) != 0) { perror(path); return -1; }
		freed += (unsigned long long)st.st_size - hl; n++;
	}
	if (!quiet) fprintf(stderr, "prune: %llu blocks cut to their heads, %.1f MB freed; the last %llu stay whole\n", n, freed / 1e6, (unsigned long long)keep);
	return 0;
}

int ch_start(const char *dir) {
	ch_config c; ch_ids d = {0};
	char path[PATHN], gp[PATHN];
	if (pj(path, dir, "tip") || access(path, F_OK) == 0) return 0;
	if (ch_load_config(dir, &c) || ids_open(dir, &d)) { ids_close(&d); return -1; }
	int bad = check_genesis(dir, &c, &d);
	ids_close(&d);
	if (bad || pj(gp, dir, "genesis.mlf2") || pj(path, dir, "tip.mlf2")) return -1;
	size_t len;
	unsigned char *b = slurp_hash(gp, &len, NULL);
	part pt = { b, len };
	int rc = !b || write_hashed(path, &pt, 1, NULL) || write_tip(dir, 0, c.genesis) ? -1 : 0;
	free(b);
	return rc;
}

/* ── over a network: what a coordinator takes in and gives out ───────────── */

int ch_submit(const char *dir, const char *file, char hex[65], const char **why) {
	ch_config c;
	uint64_t height; unsigned char parent[32], hash[32];
	size_t len;
	*why = "the chain is unreadable here";
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, parent)) return -1;
	unsigned char *b = slurp_commit(file, &len, hash);
	if (!b) { *why = "unreadable"; return -1; }
	free(b);
	tr_model *m = load_tip(dir, NULL, NULL, NULL);
	int rc = -1, nu = m ? job_size(m, &c.p) : -1;
	uint32_t *un = nu > 0 ? malloc((size_t)nu * sizeof *un) : NULL;
	char pd[PATHN], path[PATHN], name[96];
	DIR *dd = NULL;
	ch_dhead h;
	float *dl = un ? read_delta(file, hash, parent, m, &c.p, height + 1, un, &nu, &h, NULL, why) : NULL;
	if (!dl) goto out;
	free(dl);
	/* one delta a key a round: the seal would set a second aside anyway, this keeps it off the disk */
	if (pj(pd, dir, "pending") || !(dd = opendir(pd))) { *why = "no pending directory"; goto out; }
	struct dirent *e;
	while ((e = readdir(dd))) {
		if (strlen(e->d_name) != 68 || strcmp(e->d_name + 64, ".dlt")) continue;
		ch_dhead o;
		snprintf(name, sizeof name, "pending/%s", e->d_name);
		if (pj(path, dir, name)) continue;
		FILE *f = fopen(path, "rb");
		int got = f && fread(&o, sizeof o, 1, f) == 1;
		if (f) fclose(f);
		if (got && !memcmp(o.parent, parent, 32) && !memcmp(o.key, h.key, 32)) { *why = "this key already has a delta in this round"; goto out; }
	}
	sha256_hex(hash, hex);
	snprintf(name, sizeof name, "pending/%s.dlt", hex);
	if (pj(path, dir, name) || rename(file, path) != 0) { *why = "could not keep it"; goto out; }
	rc = 0;
out:
	if (dd) closedir(dd);
	free(un); tr_free(m);
	return rc;
}

int ch_status_json(const char *dir, FILE *out, int last) {
	ch_config c;
	uint64_t height; unsigned char tip[32];
	char hx[65], gx[65], path[PATHN];
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, tip)) return -1;
	sha256_hex(tip, hx); sha256_hex(c.genesis, gx);
	int pending = 0;
	DIR *dd = pj(path, dir, "pending") ? NULL : opendir(path);
	struct dirent *e;
	while (dd && (e = readdir(dd))) pending += strlen(e->d_name) == 68 && !strcmp(e->d_name + 64, ".dlt");
	if (dd) closedir(dd);
	double nd = 0;
	if (next_difficulty(dir, &c, height, &nd)) return -1;
	fprintf(out, "{\"height\":%llu,\"tip\":\"%s\",\"genesis\":\"%s\",\"pending\":%d,\"next\":{\"kind\":\"%s\",\"difficulty\":%.9g},"
	             "\"shards\":%d,\"steps\":%d,\"batch\":%d,\"experts\":%d,\"trunk\":%d,\"blocks\":[",
	        (unsigned long long)height, hx, gx, pending, c.p.experts ? trunk_round(&c.p, height + 1) ? "trunk" : "expert" : "dense", nd,
	        c.p.shards, c.p.steps, c.p.batch, c.p.experts, c.p.trunk);
	uint64_t lo = last > 0 && height > (uint64_t)last ? height - (uint64_t)last : 0;
	for (uint64_t b = height; b > lo; b--) {   /* newest first */
		unsigned char *buf; ch_bhead *hd; ch_contrib *cs; const uint32_t *bu; uint32_t nb; unsigned char hash[32];
		if (block_path(dir, b, path) || read_block(path, c.p.experts > 0, &buf, &hd, &cs, &bu, &nb, NULL, hash)) return -1;
		sha256_hex(hash, hx);
		fprintf(out, "%s{\"height\":%llu,\"hash\":\"%s\",\"kind\":\"%s\",\"difficulty\":%.9g,\"parent_loss\":%.6f,\"loss\":%.6f,\"bytes\":%llu,\"trainers\":[",
		        b == height ? "" : ",", (unsigned long long)b, hx, c.p.experts ? trunk_round(&c.p, b) ? "trunk" : "expert" : "dense",
		        hd->difficulty, hd->parent_loss, hd->new_loss, (unsigned long long)(hd->n * 4));
		for (uint32_t i = 0, w = 0; i < hd->ncontrib; i++) {   /* names are a-z A-Z 0-9 - _ . only when name_ok: anything else is left out */
			if (!memchr(cs[i].trainer, 0, CH_NAME) || !name_ok(cs[i].trainer)) continue;
			char kx[65];
			sha256_hex(cs[i].key, kx);
			fprintf(out, "%s{\"name\":\"%s\",\"key\":\"%s\",\"loss\":%.6f,\"state\":\"%s\"}", w++ ? "," : "", cs[i].trainer, kx, cs[i].loss,
			        cs[i].state == CH_MERGED ? "merged" : cs[i].state == CH_PASSED ? "passed" : "refused");
		}
		fprintf(out, "]}");
		free(buf);
	}
	fprintf(out, "]}\n");
	return ferror(out) ? -1 : 0;
}

/* ── credit ─────────────────────────────────────────────────────────────── */

/* one job's operations: 6 x the weights each token passes through x its tokens.
 * Every job on a chain runs its tokens through the same weights, whatever it was
 * dealt to train (an expert job, a trunk round, a dense job), so one number fits all */
static double job_ops(const tr_model *m, const ch_params *cp) {
	const tr_config *mc = tr_cfg(m);
	double w = mc->experts ? (double)tr_expert_off(m, 0, 0) + (double)mc->layers * mc->top_k * 3.0 * mc->ffn * mc->dim
	                       : (double)tr_n_params(m);
	return 6.0 * w * cp->steps * cp->batch * mc->ctx;
}

/* a key's entry, made when it has none: open addressing, half full at most */
static ch_credit *credit_of(ch_credit **tab, size_t *cap, size_t *n, const unsigned char key[32]) {
	#define USED(e) ((e).passed + (e).refused > 0)
	if ((*n + 1) * 2 > *cap) {
		size_t nc = *cap ? *cap * 2 : 64;
		ch_credit *nt = calloc(nc, sizeof *nt);
		if (!nt) return NULL;
		for (size_t i = 0; i < *cap; i++) {
			if (!USED((*tab)[i])) continue;
			size_t j = h64((*tab)[i].key) & (nc - 1);
			while (USED(nt[j])) j = (j + 1) & (nc - 1);
			nt[j] = (*tab)[i];
		}
		free(*tab); *tab = nt; *cap = nc;
	}
	size_t j = h64(key) & (*cap - 1);
	while (USED((*tab)[j]) && memcmp((*tab)[j].key, key, 32)) j = (j + 1) & (*cap - 1);
	if (!USED((*tab)[j])) { memcpy((*tab)[j].key, key, 32); (*n)++; }
	return &(*tab)[j];
	#undef USED
}
static int cmp_credit(const void *a, const void *b) {
	const ch_credit *x = a, *y = b;
	if (x->passed != y->passed) return x->passed < y->passed ? 1 : -1;   /* ops is passed x one job's */
	return memcmp(x->key, y->key, 32);
}

long ch_credits(const char *dir, ch_credit **out, double *job) {
	ch_config c;
	uint64_t height; unsigned char tip[32];
	*out = NULL;
	if (ch_load_config(dir, &c) || ch_tip(dir, &height, tip)) return -1;
	tr_model *m = load_tip(dir, NULL, NULL, NULL);
	if (!m) return -1;
	double jo = job_ops(m, &c.p);
	tr_free(m);
	ch_credit *tab = NULL;
	size_t cap = 0, n = 0;
	char path[PATHN];
	for (uint64_t b = 1; b <= height; b++) {
		unsigned char *buf; ch_bhead *hd; ch_contrib *cs; const uint32_t *bu; uint32_t nb; unsigned char hash[32];
		if (block_path(dir, b, path) || read_block(path, c.p.experts > 0, &buf, &hd, &cs, &bu, &nb, NULL, hash)) { free(tab); return -1; }
		for (uint32_t i = 0; i < hd->ncontrib; i++) {
			const ch_contrib *x = &cs[i];
			int named = memchr(x->trainer, 0, CH_NAME) && name_ok(x->trainer);
			if (x->state != CH_REJECTED && x->state != CH_PASSED && x->state != CH_MERGED) continue;
			ch_credit *e = credit_of(&tab, &cap, &n, x->key);
			if (!e) { free(buf); free(tab); return -1; }
			if (named) memcpy(e->trainer, x->trainer, CH_NAME);   /* blocks in order: the last label wins */
			if (x->state == CH_REJECTED) e->refused++;
			else { e->passed++; e->merged += x->state == CH_MERGED; }
		}
		free(buf);
	}
	size_t w = 0;   /* packed to the front, then sorted */
	for (size_t i = 0; i < cap; i++) if (tab[i].passed + tab[i].refused > 0) { tab[w] = tab[i]; tab[w].ops = (double)tab[w].passed * jo; w++; }
	if (w) qsort(tab, w, sizeof *tab, cmp_credit);
	*out = tab;
	if (job) *job = jo;
	return (long)w;
}

int ch_grade(double ops, const char **name, double *next) {
	static const char *const names[CH_GRADES] = { "Corax", "Nymphus", "Miles", "Leo", "Perses", "Heliodromus", "Pater" };
	int g = 0;
	double bar = CH_GRADE_FIRST;   /* 1e15 x 10^k is exact in a double up to 1e22 */
	while (g < CH_GRADES && ops >= bar) { g++; bar *= 10; }
	if (name) *name = g ? names[g - 1] : NULL;
	if (next) *next = g < CH_GRADES ? bar : 0;
	return g;
}
