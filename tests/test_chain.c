/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_chain.c — the chain: sha256 against the FIPS vectors, the shard, seed and
   retarget rules, then a tiny chain mined by two miners, sealed, replayed bit for
   bit with every loss re-run, and every bad delta and every tampered file refused.
   Then an expert chain: the units each job is dealt, jobs that move their experts
   and nothing else, merges per unit (one carrier bit for bit, two averaged), forged
   unit lists refused and a re-tipped block with wrong units caught. ASan. */
#include "../src/chain.h"
#include "../src/sha256.h"
#include "train.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

static int hash_is(const void *d, size_t n, const char *want) {
	unsigned char h[32]; char hx[65];
	sha256(d, n, h); sha256_hex(h, hx);
	return !strcmp(hx, want);
}

static const char *TEXT =
	"the box sat on the shelf and the box was old. the old box held a key, and the key opened "
	"the door at the end of the hall. nobody knew who made the box, or the key, or the door.\n"
	"she took the key from the box and walked to the door. the door was locked, the key turned, "
	"and the hall was quiet again. the box stayed on the shelf, older than before.\n\n";

static char DIRP[256];
static void at(char *out, const char *name) { snprintf(out, 512, "%s/%s", DIRP, name); }

static int count_suffix(const char *sub, const char *suffix) {
	char p[512]; at(p, sub);
	DIR *d = opendir(p);
	int n = 0;
	struct dirent *e;
	while (d && (e = readdir(d))) {
		size_t l = strlen(e->d_name), s = strlen(suffix);
		n += l > s && !strcmp(e->d_name + l - s, suffix);
	}
	if (d) closedir(d);
	return n;
}

static void rm_tree(const char *sub) {
	char p[512]; at(p, sub);
	DIR *d = opendir(p);
	struct dirent *e;
	while (d && (e = readdir(d))) {
		if (e->d_name[0] == '.' && (!e->d_name[1] || (e->d_name[1] == '.' && !e->d_name[2]))) continue;
		char f[1024]; snprintf(f, sizeof f, "%s/%s", p, e->d_name);
		remove(f);
	}
	if (d) closedir(d);
	rmdir(p);
}

/* a delta a dishonest or careless miner might send: header fields as given, then (version
   2 when nw >= 0) nw words of unit list as given, nunits first, then np floats = fill */
static void forge_units(const unsigned char parent[32], const char *miner, uint32_t shard, const uint32_t *list, int nw,
                        size_t np, float fill, int misname) {
	ch_dhead h;
	memset(&h, 0, sizeof h);
	memcpy(h.magic, "HKDL", 4); h.version = nw < 0 ? CH_VERSION : CH_UNITS;
	memcpy(h.parent, parent, 32); h.n = np; h.shard = shard; h.steps = 1;
	memcpy(h.miner, miner, strlen(miner));
	size_t ul = nw < 0 ? 0 : 4 * (size_t)nw, len = sizeof h + ul + np * 4;
	unsigned char *b = malloc(len);
	memcpy(b, &h, sizeof h);
	if (ul) memcpy(b + sizeof h, list, ul);
	float *dl = (float *)(b + sizeof h + ul);
	unsigned s = 12345;
	for (size_t j = 0; j < np; j++) { s = s * 1103515245u + 12345u; dl[j] = fill * ((float)(s >> 16 & 0x7fff) / 16384.0f - 1.0f); }
	unsigned char hash[32]; char hx[65], name[128], p[512];
	sha256(b, len, hash);
	if (misname) hash[0] ^= 1;
	sha256_hex(hash, hx);
	snprintf(name, sizeof name, "pending/%s.dlt", hx);
	at(p, name);
	FILE *f = fopen(p, "wb");
	fwrite(b, 1, len, f);
	fclose(f);
	free(b);
}
static void forge(const unsigned char parent[32], const char *miner, uint32_t shard, size_t np, float fill, int misname) {
	forge_units(parent, miner, shard, NULL, -1, np, fill, misname);
}

/* a chain file in and out whole */
static unsigned char *slurp(const char *name, size_t *len) {
	char p[512]; at(p, name);
	FILE *f = fopen(p, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	unsigned char *b = malloc((size_t)n + 1);
	*len = fread(b, 1, (size_t)n, f);
	b[*len] = 0;
	fclose(f);
	return b;
}
static void spit(const char *name, const void *b, size_t len) {
	char p[512]; at(p, name);
	FILE *f = fopen(p, "wb");
	if (f) { fwrite(b, 1, len, f); fclose(f); }
}

static void flip(const char *name, long off) {
	char p[512]; at(p, name);
	FILE *f = fopen(p, "r+b");
	if (!f) return;
	fseek(f, off, off < 0 ? SEEK_END : SEEK_SET);
	int c = fgetc(f);
	fseek(f, -1, SEEK_CUR);
	fputc(c ^ 0x10, f);
	fclose(f);
}

/* a pending delta on an expert chain: its units into un, its floats malloc'd */
static float *read_pending(const char *hex, ch_dhead *h, uint32_t *un, int maxu, int *nu) {
	char name[128];
	size_t len;
	snprintf(name, sizeof name, "pending/%s.dlt", hex);
	unsigned char *b = slurp(name, &len);
	uint32_t n = 0;
	if (!b || len < sizeof *h + 4) { free(b); return NULL; }
	memcpy(h, b, sizeof *h); memcpy(&n, b + sizeof *h, 4);
	size_t at_ = sizeof *h + 4 + 4 * (size_t)n;
	if ((int)n > maxu || len != at_ + h->n * 4) { free(b); return NULL; }
	memcpy(un, b + sizeof *h + 4, 4 * (size_t)n);
	*nu = (int)n;
	float *d = malloc(h->n * 4 + 1);
	memcpy(d, b + at_, h->n * 4);
	free(b);
	return d;
}

/* the tip a merge of deltas onto par must give, bit for bit: a unit one delta carries
   gets its floats as they are, one two carry gets their mean, the rest stays par's */
typedef struct { const uint32_t *u; int n; const float *d; } tdelta;
static int tip_is(tr_model *tip, tr_model *par, const tdelta *ds, int nds) {
	const tr_config *c = tr_cfg(par);
	size_t np = tr_n_params(par), ex = (size_t)3 * c->ffn * c->dim;
	float *want = malloc(np * 4);
	memcpy(want, tr_params(par), np * 4);
	const float *pp = tr_params(par);
	for (uint32_t u = 1; u <= (uint32_t)(c->layers * c->experts); u++) {
		const float *d[2] = { 0 }; int k = 0;
		for (int i = 0; i < nds; i++)
			for (int q = 0; q < ds[i].n; q++) if (ds[i].u[q] == u && k < 2) d[k++] = ds[i].d + (size_t)q * ex;
		size_t off = tr_expert_off(par, (int)((u - 1) / c->experts), (int)((u - 1) % c->experts));
		for (size_t j = 0; k && j < ex; j++) want[off + j] = pp[off + j] + (k == 1 ? d[0][j] : (d[0][j] + d[1][j]) / 2.0f);
	}
	int ok = tr_n_params(tip) == np && !memcmp(tr_params(tip), want, np * 4);
	free(want);
	return ok;
}

static void rm_chain(void) {
	rm_tree("blocks"); rm_tree("pending"); rm_tree(".");
	rm_tree("miners");
	rmdir(DIRP);
}

static int read_head(uint64_t h, ch_bhead *out, ch_contrib *cs, int maxc) {
	char name[64], p[512];
	snprintf(name, sizeof name, "blocks/%06llu.blk", (unsigned long long)h);
	at(p, name);
	FILE *f = fopen(p, "rb");
	if (!f) return -1;
	int ok = fread(out, sizeof *out, 1, f) == 1 && (int)out->ncontrib <= maxc &&
	         fread(cs, sizeof *cs, out->ncontrib, f) == out->ncontrib;
	fclose(f);
	return ok ? 0 : -1;
}

int main(void) {
	printf("test_chain\n");

	/* ── sha256 ── */
	ck(hash_is("", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "sha256 of nothing");
	ck(hash_is("abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "sha256 of abc");
	const char *two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
	ck(hash_is(two, strlen(two), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"), "sha256 across two blocks");
	char *ma = malloc(1000000);
	memset(ma, 'a', 1000000);
	ck(hash_is(ma, 1000000, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"), "sha256 of a million a's");
	{
		sha256_ctx c; unsigned char h1[32], h2[32];
		sha256_init(&c);
		for (size_t at_ = 0, k = 1; at_ < 1000000; at_ += k, k = k * 7 % 131 + 1)
			sha256_add(&c, ma + at_, at_ + k > 1000000 ? 1000000 - at_ : k);
		sha256_done(&c, h1);
		sha256(ma, 1000000, h2);
		ck(!memcmp(h1, h2, 32), "fed in ragged pieces, the same hash");
	}
	free(ma);

	/* ── the rules both sides compute ── */
	unsigned char par[32] = { 1, 2, 3 }, other[32] = { 9 };
	ck(ch_shard(par, "alice", 64) == ch_shard(par, "alice", 64), "a miner's shard is fixed by the parent");
	{
		int seen[64] = {0}, distinct = 0;
		char nm[16];
		for (int i = 0; i < 200; i++) { snprintf(nm, sizeof nm, "m%d", i); seen[ch_shard(par, nm, 64)] = 1; }
		for (int i = 0; i < 64; i++) distinct += seen[i];
		ck(distinct > 48, "miners spread over the shards");
		int moved = 0;
		for (int i = 0; i < 50; i++) { snprintf(nm, sizeof nm, "m%d", i); moved += ch_shard(par, nm, 64) != ch_shard(other, nm, 64); }
		ck(moved > 40, "a new parent deals new shards");
	}
	{
		unsigned char cm[3][32] = { { 5 }, { 1 }, { 3 } }, rv[3][32] = { { 3 }, { 5 }, { 1 } }, s1[32], s2[32], s3[32];
		ch_seed(par, (const unsigned char (*)[32])cm, 3, s1);
		ch_seed(par, (const unsigned char (*)[32])rv, 3, s2);
		ch_seed(par, (const unsigned char (*)[32])cm, 2, s3);
		ck(!memcmp(s1, s2, 32), "the seed does not depend on commit order");
		ck(memcmp(s1, s3, 32) != 0, "every commit moves the seed");
	}
	ck(ch_retarget(1.0, 4, 3) == 1.25 && ch_retarget(1.0, 4, 1) == 0.8 && ch_retarget(1.0, 4, 2) == 1.0 &&
	   ch_retarget(1.0, 0, 0) == 1.0, "difficulty: up past half passing, down under, flat at half or empty");
	ck(ch_retarget(1e-6, 4, 0) == 1e-6, "difficulty has a floor");
	{	/* a shard is stripes dealt round-robin: its windows reach the whole corpus, each inside a stripe it owns */
		long ntok = 100000, lo = ntok, hi = 0;
		int own = 1, fits = 1;
		for (uint64_t r = 0; r < 4000; r++) {
			long w = ch_window(ntok, 8, 64, 3, r * 0x9E3779B97F4A7C15ULL + 1, 32);
			own &= (w / 64) % 8 == 3;
			fits &= w % 64 + 33 <= 64 && w + 33 <= ntok;
			if (w < lo) lo = w;
			if (w > hi) hi = w;
		}
		ck(own && fits, "a shard's windows sit inside stripes that shard owns");
		ck(lo < ntok / 10 && hi > ntok * 9 / 10, "a shard samples both ends of the corpus, not one stretch");
	}

	/* ── a tiny chain ── */
	mkdir("build", 0755);
	snprintf(DIRP, sizeof DIRP, "build/test_chain-%ld", (long)getpid());
	char text_path[512], model_path[512];
	snprintf(text_path, sizeof text_path, "%s.txt", DIRP);
	snprintf(model_path, sizeof model_path, "%s.mlf2", DIRP);
	char *text = malloc(200 * strlen(TEXT) + 1);
	text[0] = 0;
	for (int i = 0; i < 200; i++) strcat(text, TEXT);
	FILE *tf = fopen(text_path, "wb");
	fwrite(text, 1, strlen(text), tf);
	fclose(tf);
	void *blob; size_t blen; int eos;
	int nm = bpe_train(text, strlen(text), 320, &blob, &blen, &eos);
	tr_config cfg = { 320, 32, 2, 2, 2, 16, 96, 32, 10000.0f, 1e-5f, 0, 0, 0 };
	tr_model *m0 = nm > 0 ? tr_new(&cfg, 3) : NULL;
	ck(m0 && !tr_save(m0, model_path, blob, blen, eos), "a tiny model to start from");
	char opt_path[600];
	snprintf(opt_path, sizeof opt_path, "%s.opt", model_path);
	if (m0) *tr_step(m0) = 7;   /* an optimizer that has run 7 steps */
	ck(m0 && !tr_save_opt(m0, opt_path, 1), "and its optimizer");
	size_t np = m0 ? tr_n_params(m0) : 0;
	tr_free(m0); free(text);

	ch_params p = { 8, 30, 4, 3e-3f, 8, 1e-4, 64, 0, 0 };
	ck(ch_init(DIRP, model_path, opt_path, text_path, &p) == 0, "init makes a chain");
	{
		size_t cl; unsigned char *cf = slurp("chain.cfg", &cl);
		ck(cf && cl && !strstr((char *)cf, "experts"), "a dense chain's config, and so its genesis hash, has no experts line");
		free(cf);
	}
	ck(ch_init(DIRP, model_path, opt_path, text_path, &p) != 0, "init refuses a directory that is a chain already");
	uint64_t h; unsigned char g[32], tip[32];
	ck(ch_tip(DIRP, &h, g) == 0 && h == 0, "the tip starts at genesis");
	ck(ch_verify(DIRP, 1) == 0, "genesis verifies");
	ck(ch_seal(DIRP) == 1, "seal with nothing pending writes no block");

	char ca[65], cb[65];
	ck(ch_mine(DIRP, "alice", ca) == 0 && ch_mine(DIRP, "bob", cb) == 0, "two miners mine");
	ck(strcmp(ca, cb) != 0 && count_suffix("pending", ".dlt") == 2, "two commits waiting");
	ck(ch_mine(DIRP, "bad name", ca) != 0 && ch_mine(DIRP, "", ca) != 0, "a miner name with a space, or none, is refused");
	ck(ch_seal(DIRP) == 0, "seal writes block 1");
	ch_bhead bh; ch_contrib cs[8];
	ck(ch_tip(DIRP, &h, tip) == 0 && h == 1 && read_head(1, &bh, cs, 8) == 0, "the tip moves to block 1");
	ck(!memcmp(bh.parent, g, 32), "block 1 points at genesis");
	ck(bh.ncontrib == 2 && bh.nmerged >= 1 && bh.n == np, "block 1 records both miners and carries a delta");
	ck(bh.new_loss < bh.parent_loss - bh.difficulty, "block 1 cuts the loss by the difficulty");
	ck(count_suffix("pending", ".dlt") == 0, "sealed deltas leave pending");
	ck(ch_verify(DIRP, 1) == 0, "the chain replays with every loss reproduced");

	/* round 2: one honest miner, one sending noise, and three that break the rules */
	ck(ch_mine(DIRP, "carol", ca) == 0, "carol mines block 2");
	forge(tip, "mallory", ch_shard(tip, "mallory", p.shards), np, 0.05f, 0);       /* noise: lands, fails */
	forge(g, "stale", ch_shard(g, "stale", p.shards), np, 0.0f, 0);                /* mined on genesis */
	forge(tip, "liar", (ch_shard(tip, "liar", p.shards) + 1) % p.shards, np, 0.0f, 0);   /* claims another shard */
	forge(tip, "misnamed", ch_shard(tip, "misnamed", p.shards), np, 0.0f, 1);      /* name is not its hash */
	int sealed2 = ch_seal(DIRP) == 0 && read_head(2, &bh, cs, 8) == 0;
	ck(sealed2, "seal writes block 2");
	int carol = -1, mallory = -1;
	for (uint32_t i = 0; sealed2 && i < bh.ncontrib; i++) {
		if (!strcmp(cs[i].miner, "carol")) carol = (int)i;
		if (!strcmp(cs[i].miner, "mallory")) mallory = (int)i;
	}
	ck(bh.ncontrib == 2 && carol >= 0 && mallory >= 0, "block 2 records carol and mallory, nobody else");
	ck(carol >= 0 && cs[carol].state == CH_MERGED, "carol's delta is in block 2");
	ck(mallory >= 0 && cs[mallory].state == CH_REJECTED && cs[mallory].loss > bh.parent_loss, "mallory's noise is recorded and refused");
	ck(count_suffix("pending", ".stale") == 1, "a delta on an old block is set aside as stale");
	ck(count_suffix("pending", ".bad") == 2, "a wrong shard and a wrong name are set aside as bad");
	ck(bh.difficulty == ch_retarget(p.difficulty, 2, 2), "block 2's difficulty retargets from block 1");
	ck(ch_verify(DIRP, 1) == 0, "two blocks replay with every loss reproduced");

	/* the optimizer each miner carries: genesis's for a first job, its own after */
	{
		char op[512];
		at(op, "miners/carol.opt");
		tr_model *tm = tr_load(model_path, NULL, NULL, NULL);
		ck(tm && !tr_load_opt(tm, op, NULL) && *tr_step(tm) == 7 + p.steps, "a first job starts from the genesis optimizer");
		ck(ch_mine(DIRP, "carol", ca) == 0 && ch_seal(DIRP) == 0, "carol mines block 3");
		ck(tm && !tr_load_opt(tm, op, NULL) && *tr_step(tm) == 7 + 2 * p.steps, "her next job starts from her own");
		tr_free(tm);
		ck(ch_verify(DIRP, 1) == 0, "three blocks replay");
	}

	/* tampering anywhere breaks the replay */
	flip("blocks/000001.blk", -5);
	ck(ch_verify(DIRP, 0) != 0, "a changed weight in block 1 breaks the chain");
	flip("blocks/000001.blk", -5);
	flip("blocks/000002.blk", (long)(sizeof(ch_bhead) + 76));   /* a contribution's loss */
	ck(ch_verify(DIRP, 0) != 0, "a changed loss in block 2 breaks the chain");
	flip("blocks/000002.blk", (long)(sizeof(ch_bhead) + 76));
	flip("ids.bin", 100);
	ck(ch_verify(DIRP, 0) != 0, "changed training data breaks the chain");
	flip("ids.bin", 100);
	flip("tip.mlf2", -9);
	ck(ch_verify(DIRP, 0) != 0, "a tip model that is not the replay is caught");
	flip("tip.mlf2", -9);
	ck(ch_verify(DIRP, 0) == 0, "put back, it holds again");
	{
		uint32_t whole[2] = { 1, 0 };
		ck(ch_tip(DIRP, &h, tip) == 0, "the dense tip");
		forge_units(tip, "expert", ch_shard(tip, "expert", p.shards), whole, 2, np, 0.0f, 0);
		ck(ch_seal(DIRP) == 1 && count_suffix("pending", ".bad") == 3, "a version 2 delta on a dense chain is set aside as bad");
	}
	rm_chain();

	/* ── an expert chain ── */
	{
		uint32_t u[16], w[16];
		ck(ch_units(par, "alice", 2, 4, 1, u) == 2 && u[0] >= 1 && u[0] <= 4 && u[1] >= 5 && u[1] <= 8,
		   "units: one expert of each layer, as 1 + l x experts + e");
		int ok = ch_units(par, "bob", 3, 8, 3, u) == 9 && ch_units(par, "bob", 3, 8, 3, w) == 9 && !memcmp(u, w, 9 * sizeof *u);
		for (int i = 0; i < 9; i++) ok &= u[i] >= 1 + 8 * (uint32_t)(i / 3) && u[i] < 9 + 8 * (uint32_t)(i / 3) && (i % 3 == 0 || u[i - 1] < u[i]);
		ck(ok, "units: per of each layer's experts, sorted and distinct, fixed by parent and miner");
		ok = ch_units(par, "all", 2, 4, 4, u) == 8;
		for (int i = 0; i < 8; i++) ok &= u[i] == (uint32_t)i + 1;
		ck(ok, "units: every expert when per is all of them");
		ck(ch_units(par, "x", 2, 4, 0, u) < 0 && ch_units(par, "x", 2, 4, 5, u) < 0, "units: per outside 1..experts is refused");
		int seen[8] = {0}, moved = 0, spread = 1;
		char nm[16];
		for (int i = 0; i < 200; i++) {
			snprintf(nm, sizeof nm, "m%d", i);
			ch_units(par, nm, 2, 4, 1, u); ch_units(other, nm, 2, 4, 1, w);
			seen[u[0] - 1]++; seen[u[1] - 1]++;
			moved += u[0] != w[0] || u[1] != w[1];
		}
		for (int i = 0; i < 8; i++) spread &= seen[i] > 25;
		ck(spread, "miners spread over every expert of every layer");
		ck(moved > 150, "a new parent deals new experts");
	}

	snprintf(DIRP, sizeof DIRP, "build/test_chain-moe-%ld", (long)getpid());
	char moe_path[512], moe_opt[600];
	snprintf(moe_path, sizeof moe_path, "%s.mlf2", DIRP);
	snprintf(moe_opt, sizeof moe_opt, "%s.opt", moe_path);
	tr_config mc = { 320, 32, 2, 2, 2, 16, 32, 32, 10000.0f, 1e-5f, 4, 2, 1 };
	tr_model *cold = tr_new(&mc, 5), *warm = tr_new(&mc, 5);
	size_t mnp = cold ? tr_n_params(cold) : 0, ex0 = cold ? tr_expert_off(cold, 0, 0) : 0, ex = (size_t)3 * mc.ffn * mc.dim;
	if (warm) {   /* moments on the trunk alone: one step at rate 0 on made-up trunk grads moves no weight */
		tr_zero_grad(warm);
		float *gw = tr_grads(warm);
		for (size_t j = 0; j < ex0; j++) gw[j] = (float)(j % 7) * 0.1f - 0.25f;
		tr_adamw(warm, 0.0f, 0.0f, 0.9f, 0.95f, 1e-8f);
		*tr_step(warm) = 7; *tr_step(cold) = 7;
	}
	ck(cold && warm && !memcmp(tr_params(cold), tr_params(warm), mnp * 4) && !tr_save(cold, moe_path, blob, blen, eos) &&
	   !tr_save_opt(warm, moe_opt, 1), "a tiny model with experts, and an optimizer warm on its trunk alone");
	ch_params pm = { 8, 30, 4, 3e-3f, 8, 1e-4, 64, 1, 0 }, pb = pm;
	ck(ch_init(DIRP, model_path, NULL, text_path, &pm) != 0, "init refuses experts on a dense model");
	pb.experts = 5;
	ck(ch_init(DIRP, moe_path, NULL, text_path, &pb) != 0, "init refuses more experts per job than a layer has");
	pb.experts = -1;
	ck(ch_init(DIRP, moe_path, NULL, text_path, &pb) != 0, "and a negative count");
	{ struct stat st; ck(stat(DIRP, &st) != 0, "a refused init leaves nothing behind"); }
	ck(ch_init(DIRP, moe_path, moe_opt, text_path, &pm) == 0, "init makes an expert chain");
	{	/* experts per job are in the genesis hash */
		size_t cl; unsigned char *cf = slurp("chain.cfg", &cl);
		char *e = cf ? strstr((char *)cf, "experts 1\n") : NULL;
		ck(e != NULL, "its config says experts 1");
		if (e) { e[8] = '2'; spit("chain.cfg", cf, cl); }
		ck(ch_verify(DIRP, 0) != 0, "a config that deals 2 experts per job is not this chain");
		if (e) { e[8] = '1'; spit("chain.cfg", cf, cl); }
		ck(ch_verify(DIRP, 0) == 0, "put back, it is");
		free(cf);
	}
	unsigned char g1[32];
	ck(ch_tip(DIRP, &h, g1) == 0 && h == 0, "the expert chain starts at genesis");

	/* frozen means frozen: the same job warm or cold on the trunk gives the same delta */
	char a1[65], a2[65], pa[65], gp[512];
	at(gp, "genesis.opt");
	uint32_t au[2], pu[2], uu[8];
	ch_units(g1, "alice", 2, 4, 1, au);
	ck(ch_mine(DIRP, "alice", a1) == 0, "alice mines her two experts");
	{
		char name[128];
		snprintf(name, sizeof name, "pending/%s.dlt", a1);
		size_t dl_len = 0; unsigned char *db = slurp(name, &dl_len);
		ck(db && dl_len == sizeof(ch_dhead) + 4 + 2 * 4 + 2 * ex * 4 && dl_len < mnp * 4,
		   "her delta is exactly her two experts, under a dense delta's size");
		ch_dhead hd; uint32_t n = 0, un2[2] = { 0 };
		if (db) { memcpy(&hd, db, sizeof hd); memcpy(&n, db + sizeof hd, 4); memcpy(un2, db + sizeof hd + 4, 8); }
		ck(db && hd.version == CH_UNITS && hd.n == 2 * ex && n == 2 && un2[0] == au[0] && un2[1] == au[1],
		   "version 2: the units the hash dealt her, then their floats");
		free(db);
		char mp[512];
		remove((at(mp, name), mp)); remove((at(mp, "miners/alice.opt"), mp));
	}
	ck(!tr_save_opt(cold, gp, 1) && ch_mine(DIRP, "alice", a2) == 0 && !strcmp(a1, a2),
	   "from an optimizer cold on the trunk, the same delta bit for bit (warm moments moved nothing frozen)");
	{	/* her optimizer: only her experts had a gradient */
		char op[512]; at(op, "miners/alice.opt");
		tr_model *pr = tr_load(moe_path, NULL, NULL, NULL);
		int okp = pr && !tr_load_opt(pr, op, NULL), still = 1, went = 1;
		float *b0 = malloc(mnp * 4 + 1);
		if (okp) {
			memcpy(b0, tr_params(pr), mnp * 4);
			tr_zero_grad(pr);
			tr_adamw(pr, 1.0f, 0.0f, 0.9f, 0.95f, 1e-8f);
			const float *q = tr_params(pr);
			still = !memcmp(q, b0, ex0 * 4);
			for (uint32_t un3 = 1; un3 <= 8; un3++) {
				size_t off = tr_expert_off(pr, (int)(un3 - 1) / 4, (int)(un3 - 1) % 4);
				int mine = un3 == au[0] || un3 == au[1], same = !memcmp(q + off, b0 + off, ex * 4);
				if (mine) went &= !same; else still &= same;
			}
		}
		ck(okp && still && went, "after the job only her experts carry optimizer moments: nothing else had a gradient");
		free(b0); tr_free(pr);
	}
	ck(!tr_save_opt(warm, gp, 1), "the warm optimizer back for everyone else");

	/* two miners dealt different experts: the block is each one's own, bit for bit */
	char pn[16] = "";
	for (int i = 0; i < 100; i++) {
		snprintf(pn, sizeof pn, "m%d", i);
		ch_units(g1, pn, 2, 4, 1, pu);
		if (pu[0] != au[0] && pu[1] != au[1]) break;
	}
	ck(pu[0] != au[0] && pu[1] != au[1] && ch_mine(DIRP, pn, pa) == 0, "a miner dealt other experts mines too");
	ch_dhead dh;
	uint32_t ua[2], up[2];
	int na = 0, npu = 0;
	float *da = read_pending(a2, &dh, ua, 2, &na), *dp = read_pending(pa, &dh, up, 2, &npu);
	tr_model *pmod = tr_load(moe_path, NULL, NULL, NULL), *tm = NULL;
	ck(ch_seal(DIRP) == 0 && read_head(1, &bh, cs, 8) == 0 && bh.version == CH_UNITS && bh.ncontrib == 2 && bh.nmerged == 2 &&
	   bh.n == 4 * ex, "seal merges both into block 1, four experts' floats");
	{
		char bp[512]; at(bp, "tip.mlf2");
		tm = tr_load(bp, NULL, NULL, NULL);
		tdelta ds[2] = { { ua, na, da }, { up, npu, dp } };
		ck(tm && pmod && !memcmp(tr_params(tm), tr_params(pmod), ex0 * 4), "the trunk, routers and all, is the parent's bit for bit");
		ck(tm && pmod && da && dp && tip_is(tm, pmod, ds, 2), "each expert is the parent plus its one miner's delta, bit for bit");
		size_t bl; unsigned char *bb = slurp("blocks/000001.blk", &bl);
		uint32_t n = 0;
		size_t uo = sizeof(ch_bhead) + 2 * sizeof(ch_contrib);
		if (bb && bl >= uo + 4 + 16) { memcpy(&n, bb + uo, 4); memcpy(uu, bb + uo + 4, 16); }
		int sorted = n == 4 && uu[0] < uu[1] && uu[1] < uu[2] && uu[2] < uu[3];
		for (int i = 0; sorted && i < 4; i++) sorted &= uu[i] == ua[0] || uu[i] == ua[1] || uu[i] == up[0] || uu[i] == up[1];
		ck(bb && sorted, "the block's units: the union of both miners', in order");
		free(bb);
	}
	free(da); free(dp); tr_free(pmod); tr_free(tm);
	ck(ch_verify(DIRP, 1) == 0, "the expert chain replays with every loss reproduced");

	/* round 2: two miners who share an expert, and the forgers */
	unsigned char t1[32];
	ck(ch_tip(DIRP, &h, t1) == 0 && h == 1, "the tip is block 1");
	char xn[16] = "", yn[16] = "", xc[65], yc[65];
	uint32_t xu[2] = { 0 }, yu[2] = { 0 };
	for (int i = 0, found = 0; i < 40 && !found; i++)
		for (int j = i + 1; j < 40 && !found; j++) {
			snprintf(xn, sizeof xn, "c%d", i); snprintf(yn, sizeof yn, "c%d", j);
			ch_units(t1, xn, 2, 4, 1, xu); ch_units(t1, yn, 2, 4, 1, yu);
			found = xu[0] == yu[0] && xu[1] != yu[1] && xu[1] != 8 && yu[1] != 8;   /* (1, 3) free for the tamper below */
		}
	ck(xu[0] == yu[0] && xu[1] != yu[1] && ch_mine(DIRP, xn, xc) == 0 && ch_mine(DIRP, yn, yc) == 0,
	   "two miners dealt the same expert of layer 0 mine");
	{
		uint32_t fu[2], bad[3];
		const char *who[6] = { "stray", "trunk", "unsorted", "dupe", "count", "fair" };
		for (int i = 0; i < 6; i++) {
			ch_units(t1, who[i], 2, 4, 1, fu);
			uint32_t sh = ch_shard(t1, who[i], pm.shards);
			bad[0] = 2; bad[1] = fu[0]; bad[2] = fu[1];
			if (i == 0) bad[1] = 1 + fu[0] % 4;           /* another expert of layer 0 */
			if (i == 1) bad[1] = 0;                       /* the trunk */
			if (i == 2) { bad[1] = fu[1]; bad[2] = fu[0]; }
			if (i == 3) bad[2] = fu[0];
			if (i == 4) bad[0] = 3;                       /* the units it was dealt, a count that says otherwise */
			forge_units(t1, who[i], sh, bad, 3, 2 * ex, 0.0f, 0);
		}
		forge(t1, "dense", ch_shard(t1, "dense", pm.shards), mnp, 0.0f, 0);
	}
	uint32_t ux[2], uy[2];
	int nx = 0, ny = 0;
	float *dx = read_pending(xc, &dh, ux, 2, &nx), *dy = read_pending(yc, &dh, uy, 2, &ny);
	{ char bp[512]; at(bp, "tip.mlf2"); pmod = tr_load(bp, NULL, NULL, NULL); }   /* block 2's parent, kept for the re-tip below */
	int s2 = ch_seal(DIRP) == 0 && read_head(2, &bh, cs, 8) == 0;
	int fair = -1, merged = 0;
	for (uint32_t i = 0; s2 && i < bh.ncontrib; i++) {
		if (!strcmp(cs[i].miner, "fair")) fair = (int)i;
		merged += cs[i].state == CH_MERGED && (!strcmp(cs[i].miner, xn) || !strcmp(cs[i].miner, yn));
	}
	ck(s2 && count_suffix("pending", ".bad") == 6,
	   "refused and set aside: a stray expert, the trunk, unsorted units, a duplicate, a count that lies, a dense delta");
	ck(s2 && bh.ncontrib == 3 && fair >= 0 && cs[fair].state == CH_REJECTED, "a forged delta with the units it was dealt lands (and, all zeros, fails)");
	ck(s2 && merged == 2 && bh.nmerged == 2 && bh.n == 3 * ex, "both sharers merge: three experts in block 2");
	{
		char bp[512]; at(bp, "tip.mlf2");
		tm = tr_load(bp, NULL, NULL, NULL);
		tdelta ds[2] = { { ux, nx, dx }, { uy, ny, dy } };
		ck(tm && pmod && !memcmp(tr_params(tm), tr_params(pmod), ex0 * 4), "the trunk is still the parent's");
		ck(tm && pmod && dx && dy && tip_is(tm, pmod, ds, 2), "the shared expert is the mean of both deltas, the others each one's own");
		tr_free(tm);
	}
	free(dx); free(dy);
	ck(ch_verify(DIRP, 1) == 0, "two expert blocks replay with every loss reproduced");

	/* a block whose units are not its miners', re-tipped so every hash agrees: verify still says no */
	{
		size_t bl, tl, ml;
		unsigned char *bb = slurp("blocks/000002.blk", &bl), *tb = slurp("tip", &tl), *mb = slurp("tip.mlf2", &ml);
		size_t hl = sizeof(ch_bhead) + 3 * sizeof(ch_contrib);   /* head and contributions */
		uint32_t n = 0, ul0[3] = { 0 };
		int shape = bb && tb && mb && bl == hl + 4 + 12 + 3 * ex * 4;
		if (shape) { memcpy(&n, bb + hl, 4); memcpy(ul0, bb + hl + 4, 12); shape = n == 3 && ul0[0] >= 1 && ul0[0] <= 4 && ul0[2] < 8; }
		ck(shape, "block 2 as it should be: three units, the first of layer 0, none of them expert (1, 3)");
		const char *says[4] = { "a block naming an expert no merged miner was dealt breaks the chain, though re-tipped",
		                        "a block naming the trunk breaks the chain",
		                        "a block with one expert more than its miners' breaks the chain, though re-tipped",
		                        "a block with more floats than its experts take breaks the chain" };
		for (int kind = 0; shape && kind < 4; kind++) {
			uint32_t ul[4] = { ul0[0], ul0[1], ul0[2], 8 };   /* kind 2: expert (1, 3), nobody's, after the rest */
			int nk = kind == 2 ? 4 : 3, nf = kind >= 2 ? 4 : 3;
			if (kind == 0) ul[0] = 1 + ul0[0] % 4;            /* another expert of layer 0 */
			if (kind == 1) ul[0] = 0;                         /* the trunk */
			size_t tl2 = hl + 4 + 4 * (size_t)nk + (size_t)nf * ex * 4;
			unsigned char *t = calloc(1, tl2);
			float *f = (float *)(t + hl + 4 + 4 * (size_t)nk);
			memcpy(t, bb, hl); ((ch_bhead *)t)->n = (uint64_t)nf * ex;
			memcpy(t + hl, &(uint32_t){ (uint32_t)nk }, 4); memcpy(t + hl + 4, ul, 4 * (size_t)nk);
			memcpy(f, bb + hl + 16, 3 * ex * 4);
			for (size_t j = 0; nf == 4 && j < ex; j++) f[3 * ex + j] = nk == 4 ? 0.01f : 0.0f;
			spit("blocks/000002.blk", t, tl2); spit("tip.mlf2", mb, ml);   /* the honest tip model unless rebuilt below */
			unsigned char hh[32]; char hx[65], line[96];
			sha256(t, tl2, hh); sha256_hex(hh, hx);
			snprintf(line, sizeof line, "2 %s\n", hx);
			spit("tip", line, strlen(line));
			if (kind != 1 && kind != 3 && pmod) {   /* the tip model too: the parent plus the block's floats where it now says */
				void *tb2 = NULL; size_t tbl; int te; char bp[512];
				at(bp, "tip.mlf2");
				tr_model *r = tr_load(bp, &tb2, &tbl, &te);
				float *q = r ? tr_params(r) : NULL;
				if (q) {
					memcpy(q, tr_params(pmod), mnp * 4);
					for (int k = 0; k < nk; k++) {
						size_t off = tr_expert_off(r, (int)(ul[k] - 1) / 4, (int)(ul[k] - 1) % 4);
						for (size_t j = 0; j < ex; j++) q[off + j] += f[(size_t)k * ex + j];
					}
					tr_save(r, bp, tb2, tbl, te);
				}
				free(tb2); tr_free(r);
			}
			ck(ch_verify(DIRP, 0) != 0, says[kind]);
			free(t);
		}
		if (shape) { spit("blocks/000002.blk", bb, bl); spit("tip", tb, tl); spit("tip.mlf2", mb, ml); }
		ck(shape && ch_verify(DIRP, 1) == 0, "put back, it holds again");
		free(bb); free(tb); free(mb);
	}
	tr_free(pmod);

	/* round 3: nothing passes, so a block with no units */
	{
		unsigned char t2[32]; uint32_t fu[3] = { 2 };
		ck(ch_tip(DIRP, &h, t2) == 0 && h == 2, "the tip is block 2");
		ch_units(t2, "fair", 2, 4, 1, fu + 1);
		forge_units(t2, "fair", ch_shard(t2, "fair", pm.shards), fu, 3, 2 * ex, 0.0f, 0);
		size_t bl = 0; unsigned char *bb = NULL; uint32_t n = 1;
		int s3 = ch_seal(DIRP) == 0 && read_head(3, &bh, cs, 8) == 0 && (bb = slurp("blocks/000003.blk", &bl));
		if (bb) memcpy(&n, bb + sizeof(ch_bhead) + sizeof(ch_contrib), 4);
		ck(s3 && bh.nmerged == 0 && bh.n == 0 && n == 0 && bl == sizeof(ch_bhead) + sizeof(ch_contrib) + 4,
		   "a block where nothing passed carries no units and no floats");
		free(bb);
		ck(ch_verify(DIRP, 1) == 0, "three expert blocks replay");
	}
	rm_chain();

	/* trunk rounds: every second block trains the trunk, every job the same unit 0 */
	snprintf(DIRP, sizeof DIRP, "build/test_chain-trunk-%ld", (long)getpid());
	{
		ch_params pt = pm, pz = p;
		pz.trunk = 2;
		ck(ch_init(DIRP, model_path, NULL, text_path, &pz) != 0, "init refuses trunk rounds on a dense chain");
		pt.trunk = -1;
		ck(ch_init(DIRP, moe_path, NULL, text_path, &pt) != 0, "and a negative trunk");
		pt.trunk = 2;
		ck(ch_init(DIRP, moe_path, moe_opt, text_path, &pt) == 0, "init makes an expert chain with trunk rounds");
		size_t cl; unsigned char *cf = slurp("chain.cfg", &cl);
		char *e = cf ? strstr((char *)cf, "trunk 2\n") : NULL;
		ck(e != NULL, "its config says trunk 2");
		if (e) { e[6] = '3'; spit("chain.cfg", cf, cl); }
		ck(ch_verify(DIRP, 0) != 0, "another trunk rhythm is not this chain");
		if (e) { e[6] = '2'; spit("chain.cfg", cf, cl); }
		free(cf);
		char c1[65], c2[65];
		ck(ch_mine(DIRP, "ann", c1) == 0 && ch_seal(DIRP) == 0 && read_head(1, &bh, cs, 8) == 0 && bh.n % ex == 0,
		   "block 1 is an expert round");
		unsigned char t1[32];
		ck(ch_tip(DIRP, &h, t1) == 0 && h == 1, "the tip is block 1");
		char bp[512]; at(bp, "tip.mlf2");
		tr_model *par = tr_load(bp, NULL, NULL, NULL);
		ck(par && ch_mine(DIRP, "ann", c1) == 0 && ch_mine(DIRP, "bob", c2) == 0, "two miners mine block 2, a trunk round");
		ch_dhead h1, h2;
		uint32_t u1[2] = { 9, 9 }, u2[2] = { 9, 9 };
		int n1 = 0, n2 = 0;
		float *d1 = read_pending(c1, &h1, u1, 2, &n1), *d2 = read_pending(c2, &h2, u2, 2, &n2);
		ck(d1 && d2 && n1 == 1 && n2 == 1 && u1[0] == 0 && u2[0] == 0 && h1.n == ex0 && h2.n == ex0 && h1.version == CH_UNITS,
		   "each delta is unit 0, the trunk's floats and no expert's");
		int moved = 0;
		for (size_t j = 0; d1 && j < ex0; j++) moved += d1[j] != 0;
		ck(moved > (int)(ex0 / 2), "the trunk moved");
		{	/* forgers in a trunk round */
			uint32_t fu[2], bad[3];
			ch_units(t1, "greedy", 2, 4, 1, fu);
			bad[0] = 2; bad[1] = fu[0]; bad[2] = fu[1];
			forge_units(t1, "greedy", ch_shard(t1, "greedy", pt.shards), bad, 3, 2 * ex, 0.0f, 0);   /* the experts an expert round would deal it */
			bad[0] = 2; bad[1] = 0; bad[2] = fu[0];
			forge_units(t1, "both", ch_shard(t1, "both", pt.shards), bad, 3, ex0 + ex, 0.0f, 0);     /* the trunk and an expert */
			bad[0] = 1; bad[1] = 0;
			forge_units(t1, "fair", ch_shard(t1, "fair", pt.shards), bad, 2, ex0, 0.0f, 0);
		}
		int s2 = ch_seal(DIRP) == 0 && read_head(2, &bh, cs, 8) == 0;
		ck(s2 && count_suffix("pending", ".bad") == 2, "refused in a trunk round: a delta of experts, a delta of the trunk and an expert");
		int m1 = 0, m2 = 0, fair = -1;
		for (uint32_t i = 0; s2 && i < bh.ncontrib; i++) {
			if (!strcmp(cs[i].miner, "ann")) m1 = cs[i].state == CH_MERGED;
			if (!strcmp(cs[i].miner, "bob")) m2 = cs[i].state == CH_MERGED;
			if (!strcmp(cs[i].miner, "fair")) fair = (int)i;
		}
		ck(s2 && bh.ncontrib == 3 && fair >= 0 && cs[fair].state == CH_REJECTED, "a zero trunk delta lands and fails");
		ck(s2 && (m1 || m2) && bh.n == ex0 && bh.version == CH_UNITS, "block 2 carries the trunk");
		tr_model *tip = tr_load(bp, NULL, NULL, NULL);
		int okt = tip && par && d1 && d2, mean = 1, exs = 1;
		if (okt) {
			const float *q = tr_params(tip), *b0 = tr_params(par);
			for (size_t j = 0; j < ex0; j++) {
				float dl = m1 && m2 ? (0.0f + d1[j] + d2[j]) / 2.0f : m1 ? d1[j] : d2[j];
				float w = b0[j] + dl;
				mean &= !memcmp(&w, &q[j], 4);
			}
			exs = !memcmp(q + ex0, b0 + ex0, (mnp - ex0) * 4);
		}
		ck(okt && mean, m1 && m2 ? "the tip's trunk is the parent's plus the mean of both, bit for bit" : "the tip's trunk is the parent's plus the one that merged, bit for bit");
		ck(okt && exs, "and every expert is the parent's, bit for bit");
		free(d1); free(d2); tr_free(par); tr_free(tip);
		ck(ch_mine(DIRP, "ann", c1) == 0, "block 3 is an expert round again");
		d1 = read_pending(c1, &h1, u1, 2, &n1);
		ck(d1 && n1 == 2 && u1[0] >= 1 && h1.n == 2 * ex, "its delta is experts, no trunk");
		free(d1);
		ck(ch_seal(DIRP) == 0 && ch_verify(DIRP, 1) == 0, "the chain replays through both kinds of round, every loss reproduced");
		{	/* a trunk block passed off as carrying an expert instead */
			size_t bl; unsigned char *bb = slurp("blocks/000002.blk", &bl);
			size_t uo = sizeof(ch_bhead) + 3 * sizeof(ch_contrib) + 4;
			uint32_t one = 1;
			if (bb && bl > uo + 4) { memcpy(bb + uo, &one, 4); spit("blocks/000002.blk", bb, bl); }
			ck(bb && ch_verify(DIRP, 0) != 0, "a trunk block renamed to an expert breaks the chain");
			free(bb);
		}
	}
	rm_chain();

	/* a model with experts on a dense chain: every job trains every weight */
	snprintf(DIRP, sizeof DIRP, "build/test_chain-moed-%ld", (long)getpid());
	{
		ch_params pd = pm;
		pd.experts = 0; pd.steps = 5;
		char dc[65], name[128];
		ck(ch_init(DIRP, moe_path, NULL, text_path, &pd) == 0 && ch_mine(DIRP, "dora", dc) == 0, "a model with experts mines on a dense chain");
		snprintf(name, sizeof name, "pending/%s.dlt", dc);
		size_t dl_len = 0; unsigned char *db = slurp(name, &dl_len);
		ck(db && dl_len == sizeof(ch_dhead) + mnp * 4 && ((ch_dhead *)db)->version == CH_VERSION, "its delta is a dense one, every weight");
		free(db);
		ck(ch_seal(DIRP) == 0 && ch_verify(DIRP, 1) == 0, "and it seals and replays");
	}
	rm_chain();
	tr_free(cold); tr_free(warm); free(blob);
	remove(moe_path); remove(moe_opt);

	remove(text_path); remove(model_path); remove(opt_path);
	printf(fails ? "test_chain: %d FAILED\n" : "test_chain: all passed\n", fails);
	return fails != 0;
}
