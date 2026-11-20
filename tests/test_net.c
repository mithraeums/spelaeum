/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_net.c — a chain over HTTP, for real on localhost: a coordinator in a child
   process, a trainer that fetches the chain, replays it, trains and submits, a round
   sealed on the clock, the trainer's copy brought up to the new block and verified
   with every loss re-run. And what the coordinator must refuse: a second delta
   from one key (under any name), a signature that is not the key's, junk, an
   oversized body, a stale delta, and any file that is not on its list (pending
   deltas above all). Then a mirror: a coordinator whose mirror is down, its
   --after program, and a second chain laid out for a static host (file:// here)
   that a new copy joins from, its snapshot trailing the tip, with the blocks it
   lacks, the model it has wrong and the wait for it all falling back to the
   coordinator. Needs curl. ASan. */
#include "../src/chain.h"
#include "../src/net.h"
#include "../src/sha256.h"
#include "train.h"
#include "bpe.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int fails = 0;
/* a trainer's key, made from its name */
static const ch_key *KEY(const char *name) {
	static struct { char n[CH_NAME]; ch_key k; } ks[16];
	static int nk;
	for (int i = 0; i < nk; i++) if (!strcmp(ks[i].n, name)) return &ks[i].k;
	if (nk == 16) exit(1);
	unsigned char seed[32];
	sha256(name, strlen(name), seed);
	snprintf(ks[nk].n, sizeof ks[nk].n, "%s", name);
	ch_key_from(seed, &ks[nk].k);
	return &ks[nk++].k;
}
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

static const char *TEXT =
	"the box sat on the shelf and the box was old. the old box held a key, and the key opened "
	"the door at the end of the hall. nobody knew who made the box, or the key, or the door.\n"
	"she took the key from the box and walked to the door. the door was locked, the key turned, "
	"and the hall was quiet again. the box stayed on the shelf, older than before.\n\n";

/* the status code of one curl call; args are this test's own strings */
static int http(const char *args) {
	char cmd[1400];
	snprintf(cmd, sizeof cmd, "curl -s -o /dev/null -w '%%{http_code}' %s", args);
	FILE *p = popen(cmd, "r");
	int code = 0;
	if (p) { if (fscanf(p, "%d", &code) != 1) code = 0; pclose(p); }
	return code;
}
static long fsize(const char *path) { struct stat st; return stat(path, &st) == 0 ? (long)st.st_size : -1; }
static int same_file(const char *a, const char *b) {
	char cmd[1200];
	snprintf(cmd, sizeof cmd, "cmp -s '%s' '%s'", a, b);
	return system(cmd) == 0;
}
/* the first line of a small file, "" when there is none */
static const char *line_of(const char *path) {
	static char l[200];
	FILE *f = fopen(path, "r");
	l[0] = 0;
	if (f) { if (!fgets(l, sizeof l, f)) l[0] = 0; fclose(f); }
	l[strcspn(l, "\n")] = 0;
	return l;
}
static int count_dlt(const char *dir) {
	char cmd[700];
	snprintf(cmd, sizeof cmd, "ls '%s/pending' 2>/dev/null | grep -c '\\.dlt$'", dir);
	FILE *p = popen(cmd, "r");
	int n = -1;
	if (p) { if (fscanf(p, "%d", &n) != 1) n = -1; pclose(p); }
	return n;
}

int main(void) {
	char S[256], M[256], text_path[300], model_path[300], opt_path[320], url[64], a[1300], b[700];
	long pid = (long)getpid();
	int port = 20000 + (int)(pid % 20000);
	mkdir("build", 0755);
	setenv("SPEL_KEEP_FREE", "0", 1);   /* this machine's free disk is not the test's to need */
	snprintf(S, sizeof S, "build/test_net-%ld-coord", pid);
	snprintf(M, sizeof M, "build/test_net-%ld-trainer", pid);
	snprintf(text_path, sizeof text_path, "build/test_net-%ld.txt", pid);
	snprintf(model_path, sizeof model_path, "build/test_net-%ld.mlf2", pid);
	snprintf(opt_path, sizeof opt_path, "%s.opt", model_path);
	snprintf(url, sizeof url, "http://127.0.0.1:%d/", port);   /* the trailing slash must not matter */

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
	if (m0) *tr_step(m0) = 7;
	ck(m0 && !tr_save(m0, model_path, blob, blen, eos) && !tr_save_opt(m0, opt_path, 1), "a tiny model and its optimizer");
	tr_free(m0); free(text); free(blob);
	ch_params p = { 8, 30, 4, 3e-3f, 8, 1e-4, 64, 0, 0 };
	ck(ch_init(S, model_path, opt_path, text_path, &p) == 0, "the coordinator's chain");

	/* its mirror is down (nothing listens on port 1), and each block runs a program */
	char hook[300];
	snprintf(hook, sizeof hook, "build/test_net-%ld-after.sh", pid);
	{ FILE *hf = fopen(hook, "w"); if (hf) { fprintf(hf, "#!/bin/sh\necho \"$2\" > \"$1/after\"\n"); fclose(hf); } chmod(hook, 0755); }
	fprintf(stderr, "(the two refusals below are expected)\n");
	ck(net_serve(S, NULL, port, 4, "file:///etc", NULL, -1) != 0, "a coordinator names no file:// mirror: a disk is a trainer's own to name");
	ck(net_serve(S, NULL, port, 4, NULL, "build/no-such-program", -1) != 0, "nor runs an --after that is not there");
	int rp[2];
	if (pipe(rp) != 0) return 1;
	fflush(stdout);
	pid_t srv = fork();
	if (srv == 0) { close(rp[0]); _exit(net_serve(S, NULL, port, 8, "http://127.0.0.1:1/", hook, rp[1]) ? 1 : 0); }
	close(rp[1]);
	char rb;
	ck(srv > 0 && read(rp[0], &rb, 1) == 1, "the coordinator listens");
	/* a third, closing its rounds for checks: forked now, before this process has trained
	   anything, since a child forked after the engine's threads started has none to run on */
	char S3[300];
	snprintf(S3, sizeof S3, "build/test_net-%ld-c3", pid);
	ck(ch_init(S3, model_path, opt_path, text_path, &p) == 0, "checks: a third chain");
	net_checks cx = { 1, 0, 30 };
	int rq3[2];
	if (pipe(rq3) != 0) return 1;
	fflush(stdout);
	pid_t s3 = fork();
	if (s3 == 0) { close(rq3[0]); _exit(net_serve_checked(S3, NULL, port + 3, 8, NULL, NULL, &cx, rq3[1]) ? 1 : 0); }
	close(rq3[1]);
	ck(s3 > 0 && read(rq3[0], &rb, 1) == 1, "checks: its coordinator listens, closing rounds for checks");
	/* a fourth on the first chain, never sealing, told to keep more disk free than any machine has */
	int rq4[2];
	if (pipe(rq4) != 0) return 1;
	fflush(stdout);
	pid_t s4 = fork();
	if (s4 == 0) { close(rq4[0]); setenv("SPEL_KEEP_FREE", "1099511627776", 1); _exit(net_serve(S, NULL, port + 4, 3600, NULL, NULL, rq4[1]) ? 1 : 0); }
	close(rq4[1]);
	ck(s4 > 0 && read(rq4[0], &rb, 1) == 1, "a coordinator short of disk listens");
	char url4[64];
	snprintf(url4, sizeof url4, "http://127.0.0.1:%d", port + 4);

	net_tip t;
	ck(net_sync(url, NULL, M, &t, 1) == 0 && t.height == 0 && t.round == 8 && t.seal > t.now - 1, "a trainer fetches the chain into an empty place and replays it");
	ck(!strcmp(t.files, "http://127.0.0.1:1/") && t.from_files == 0 && t.from_url == 3 && !t.has_mirrored,
	   "the mirror the coordinator names is down: the data, genesis and its optimizer from the coordinator");
	snprintf(a, sizeof a, "%s/tip", S); snprintf(b, sizeof b, "%s/tip", M);
	ck(same_file(a, b), "its tip is the coordinator's");
	snprintf(a, sizeof a, "%s/genesis.opt", M);
	ck(fsize(a) > 0, "it has the genesis optimizer too");
	ck(net_sync(url, NULL, M, &t, 1) == 0, "a second sync with nothing new is fine");

	char ann[65], bob[65], again[65];
	ck(ch_train(M, "ann", KEY("ann"), ann) == 0 && ch_train(M, "bob", KEY("bob"), bob) == 0, "two jobs trained on the trainer's copy");
	char keep[700];
	snprintf(a, sizeof a, "%s/pending/%s.dlt", M, bob); snprintf(keep, sizeof keep, "%s/bob.keep", M);
	{ char cmd[1500]; snprintf(cmd, sizeof cmd, "cp '%s' '%s'", a, keep); ck(system(cmd) == 0 && remove(a) == 0, "bob holds his delta back for later"); }
	ck(net_submit(url4, M, ann) != 0 && count_dlt(S) == 0 && count_dlt(M) == 1, "a coordinator short of disk refuses a delta and keeps nothing; the trainer keeps it");
	kill(s4, SIGTERM);
	waitpid(s4, NULL, 0);
	ck(net_submit(url, M, ann) == 0 && count_dlt(S) == 1 && count_dlt(M) == 0, "ann's delta goes to the coordinator and leaves her machine");
	snprintf(a, sizeof a, "%s/trainers/ann.opt", M);
	ck(remove(a) == 0 && ch_train(M, "ann", KEY("ann"), again) == 0 && !strcmp(again, ann), "the same job trained again is the same delta");
	fprintf(stderr, "(the refusal below is expected)\n");
	ck(net_submit(url, M, again) != 0 && count_dlt(S) == 1, "a second delta from the same trainer in one round is refused");
	snprintf(a, sizeof a, "%s/pending/%s.dlt", M, again); remove(a);
	snprintf(a, sizeof a, "%s/trainers/anna.opt", M); remove(a);
	fprintf(stderr, "(the refusal below is expected)\n");
	ck(ch_train(M, "anna", KEY("ann"), again) == 0 && strcmp(again, ann) && net_submit(url, M, again) != 0 && count_dlt(S) == 1,
	   "nor under another name: one key, one delta a round");
	snprintf(a, sizeof a, "%s/pending/%s.dlt", M, again); remove(a);
	{	/* bob's delta with one bit of its signature flipped: same commit, not his signature */
		char cmd[1500];
		snprintf(a, sizeof a, "%s/pending/%s.dlt", M, bob);
		snprintf(cmd, sizeof cmd, "cp '%s' '%s'", keep, a);
		FILE *f = system(cmd) == 0 ? fopen(a, "r+b") : NULL;
		if (f) { fseek(f, -7, SEEK_END); int c = fgetc(f); fseek(f, -1, SEEK_CUR); fputc(c ^ 4, f); fclose(f); }
		fprintf(stderr, "(the refusal below is expected)\n");
		ck(f && net_submit(url, M, bob) != 0 && count_dlt(S) == 1, "a delta whose signature is not its key's is refused");
		remove(a);
	}

	/* what must not leave, and what must not come in */
	snprintf(a, sizeof a, "'%spending/%s.dlt'", url, ann);
	ck(http(a) == 404, "a pending delta cannot be fetched: nobody sees one before the seal");
	snprintf(a, sizeof a, "--path-as-is '%sblocks/../chain.cfg'", url);
	ck(http(a) == 404, "a path that climbs is not on the list");
	snprintf(a, sizeof a, "'%strainers/ann.opt'", url);
	ck(http(a) == 404, "nor is anything else in the directory");
	snprintf(a, sizeof a, "'%schain.cfg'", url);
	ck(http(a) == 200, "the config is");
	snprintf(a, sizeof a, "-X DELETE '%stip'", url);
	ck(http(a) == 405, "only GET and POST");
	snprintf(b, sizeof b, "build/test_net-%ld.junk", pid);
	{ FILE *f = fopen(b, "wb"); char z[1000] = { 0 }; fwrite(z, 1, sizeof z, f); fclose(f); }
	snprintf(a, sizeof a, "-H 'Expect:' --data-binary '@%s' '%ssubmit'", b, url);
	ck(http(a) == 400 && count_dlt(S) == 1, "junk is refused and not kept");
	snprintf(a, sizeof a, "%s/genesis.mlf2", S);
	{ FILE *f = fopen(b, "wb"); long n = fsize(a) + 70000; char z[4096] = { 0 }; for (long w = 0; w < n; w += sizeof z) fwrite(z, 1, sizeof z, f); fclose(f); }
	snprintf(a, sizeof a, "-H 'Expect:' --data-binary '@%s' '%ssubmit'", b, url);
	ck(http(a) == 413, "a body bigger than any delta is refused before it is read");
	remove(b);
	snprintf(b, sizeof b, "build/test_net-%ld.json", pid);
	{
		char cmd[900];
		snprintf(cmd, sizeof cmd, "curl -s -o '%s' '%sstatus.json' && grep -q '\"height\":0,' '%s' && grep -q '\"pending\":1,' '%s'", b, url, b, b);
		ck(system(cmd) == 0, "the status page's json: height 0, one delta pending");
	}

	/* the clock seals the round */
	int moved = 0;
	for (int i = 0; i < 40 && !moved; i++) { net_tip n; usleep(500000); moved = net_tip_get(url, M, &n) == 0 && n.height == 1; }
	ck(moved, "the coordinator seals on its clock: height 1");
	{
		char ap[400];
		snprintf(ap, sizeof ap, "%s/after", S);
		for (int i = 0; i < 40 && strcmp(line_of(ap), "1"); i++) usleep(100000);
		ck(!strcmp(line_of(ap), "1"), "--after ran with the chain and the new height");
	}
	ck(net_sync(url, NULL, M, &t, 1) == 0 && t.height == 1, "the trainer fetches block 1 and replays it");
	snprintf(a, sizeof a, "%s/tip.mlf2", S); snprintf(b, sizeof b, "%s/tip.mlf2", M);
	ck(same_file(a, b), "its model is the coordinator's, byte for byte");
	ck(ch_verify(M, 1) == 0, "and the trainer's copy verifies with every loss re-run");
	{	/* a trainer that starts late: heads and the model, no old delta */
		char L[300], f[700], g[700];
		snprintf(L, sizeof L, "build/test_net-%ld-late", pid);
		snprintf(a, sizeof a, "'%sheads/1'", url);
		ck(http(a) == 200, "the heads from block 1 can be fetched");
		snprintf(a, sizeof a, "'%sheads/0'", url); snprintf(b, sizeof b, "'%sheads/9'", url);
		ck(http(a) == 404 && http(b) == 404, "heads from 0, or past the tip, cannot");
		ck(ch_prune(S, 0, 0) == 0, "the coordinator prunes every block to its head");
		ck(net_sync(url, NULL, L, &t, 0) == 0 && t.height == 1, "a new trainer syncs from the heads and the model alone");
		snprintf(f, sizeof f, "%s/tip", L); snprintf(g, sizeof g, "%s/tip", S);
		ck(same_file(f, g), "its tip is the coordinator's");
		snprintf(f, sizeof f, "%s/tip.mlf2", L); snprintf(g, sizeof g, "%s/tip.mlf2", S);
		ck(same_file(f, g), "its model too, byte for byte");
		snprintf(f, sizeof f, "%s/blocks/000001.blk", L); snprintf(g, sizeof g, "%s/blocks/000001.blk", M);
		ck(fsize(f) > 0 && fsize(f) < fsize(g), "it holds block 1's head, not its delta");
		char lc[65];
		ck(ch_train(L, "cy", KEY("cy"), lc) == 0, "and trains on it");
		char cmd[400];
		snprintf(cmd, sizeof cmd, "rm -rf '%s'", L);
		if (system(cmd) != 0) fails++;
	}
	{
		char cmd[900];
		snprintf(b, sizeof b, "build/test_net-%ld.json", pid);
		snprintf(cmd, sizeof cmd, "curl -s -o '%s' '%sstatus.json' && grep -q '\"name\":\"ann\",\"key\":\"[0-9a-f]\\{64\\}\"' '%s'", b, url, b);
		ck(system(cmd) == 0, "the json names ann in block 1, with her key");
		remove(b);
	}
	snprintf(a, sizeof a, "%s/pending/%s.dlt", M, bob);
	fprintf(stderr, "(the refusal below is expected)\n");
	ck(rename(keep, a) == 0 && net_submit(url, M, bob) != 0 && count_dlt(S) == 0, "bob's delta, trained on the old tip, is refused as stale");
	{	/* a block the coordinator did not make */
		snprintf(a, sizeof a, "%s/blocks/000002.blk", M);
		char cmd[1500];
		snprintf(cmd, sizeof cmd, "cp '%s/blocks/000001.blk' '%s'", M, a);
		ck(system(cmd) == 0 && net_sync(url, NULL, M, &t, 1) != 0, "a copy holding a block the coordinator lacks is refused, not papered over");
	}

	kill(srv, SIGTERM);
	waitpid(srv, NULL, 0);
	remove(hook);

	/* ── a mirror: a second chain laid out for a static host, here file:// ── */
	{
		char S2[300], MI[300], F[300], F2[300], u2[64], abs_[2048], mu[2100], x[2200], y[2200];
		snprintf(S2, sizeof S2, "build/test_net-%ld-c2", pid);
		snprintf(MI, sizeof MI, "build/test_net-%ld-mirror", pid);
		snprintf(F, sizeof F, "build/test_net-%ld-f", pid);
		snprintf(F2, sizeof F2, "build/test_net-%ld-f2", pid);
		snprintf(u2, sizeof u2, "http://127.0.0.1:%d", port + 1);
		ck(ch_init(S2, model_path, opt_path, text_path, &p) == 0, "a second chain, to mirror");
		int rq[2];
		if (pipe(rq) != 0) return 1;
		fflush(stdout);
		pid_t s2 = fork();   /* sealed by hand here, not on its clock */
		if (s2 == 0) { close(rq[0]); _exit(net_serve(S2, NULL, port + 1, 3600, "http://127.0.0.1:1/m", NULL, rq[1]) ? 1 : 0); }
		close(rq[1]);
		ck(s2 > 0 && read(rq[0], &rb, 1) == 1, "its coordinator listens");
		char hx[65];
		int up = 1;
		for (int h = 1; up && h <= 10; h++) up = ch_train(S2, "dee", KEY("dee"), hx) == 0 && ch_seal(S2) == 0;
		ck(up && net_mirror(S2, MI, 100, NULL) == 0, "ten blocks, and the mirror laid out: every block and a snapshot at 10");
		snprintf(x, sizeof x, "%s/blocks/000001.blk", S2); snprintf(y, sizeof y, "%s/blocks/000/000001.blk", MI);
		int same = same_file(x, y);
		static const char *top[] = { "chain.cfg", "genesis.mlf2", "genesis.opt", "ids.bin", "tip", "tip.mlf2" };
		for (int i = 0; i < 6; i++) { snprintf(x, sizeof x, "%s/%s", S2, top[i]); snprintf(y, sizeof y, "%s/%s", MI, top[i]); same &= same_file(x, y); }
		ck(same, "the mirror is the chain's own files, its blocks in folders of a thousand");
		for (int h = 11; up && h <= 12; h++) up = ch_train(S2, "dee", KEY("dee"), hx) == 0 && ch_seal(S2) == 0;
		char *added = NULL; size_t al = 0;
		FILE *am = open_memstream(&added, &al);
		ck(up && am && net_mirror(S2, MI, 100, am) == 0 && fclose(am) == 0 && added &&
		   !strcmp(added, "blocks/000/000011.blk\nblocks/000/000012.blk\n"), "two blocks on: the mirror takes them, its snapshot stays at 10");
		free(added);
		snprintf(y, sizeof y, "%s/tip", MI);
		ck(!strncmp(line_of(y), "10 ", 3), "the mirror's tip says 10");
		ck(realpath(MI, abs_) != NULL, "the mirror's place");
		{	/* a URL has no spaces: this clone sits under "Mobile Documents" */
			size_t k = (size_t)snprintf(mu, sizeof mu, "file://");
			for (const char *c = abs_; *c && k + 4 < sizeof mu; c++) k += (size_t)(*c == ' ' ? snprintf(mu + k, sizeof mu - k, "%%20") : snprintf(mu + k, sizeof mu - k, "%c", *c));
			snprintf(mu + k, sizeof mu - k, "/");
		}
		ck(net_sync(u2, mu, F, &t, 0) == 0 && t.height == 12 && t.from_url == 0 && t.from_files == 6,
		   "a new copy joins at the mirror's snapshot and takes blocks 11 and 12 from it: nothing big from the coordinator");
		snprintf(x, sizeof x, "%s/tip.mlf2", S2); snprintf(y, sizeof y, "%s/tip.mlf2", F);
		ck(same_file(x, y), "its model is the coordinator's, byte for byte");
		ck(!strcmp(t.files, "http://127.0.0.1:1/m") && !t.has_mirrored, "the coordinator's own mirror is named, this machine's is used");
		/* block 13, not mirrored yet, and the coordinator says the mirror has 12 */
		up = ch_train(S2, "dee", KEY("dee"), hx) == 0 && ch_seal(S2) == 0;
		snprintf(x, sizeof x, "%s/mirrored", S2);
		{ FILE *mf = fopen(x, "w"); if (mf) { fputs("12\n", mf); fclose(mf); } }
		setenv("SPEL_MIRROR_WAIT", "1", 1);
		struct timespec t0, t1;
		clock_gettime(CLOCK_MONOTONIC, &t0);
		int ok = up && net_sync(u2, mu, F, &t, 0) == 0;
		clock_gettime(CLOCK_MONOTONIC, &t1);
		double took = (double)(t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
		ck(ok && t.height == 13 && t.has_mirrored && t.mirrored == 12, "the coordinator says how far its mirror has got");
		ck(ok && took >= 0.9 && t.from_files == 0 && t.from_url == 1, "a trainer waits for the mirror, then takes the block it lacks from the coordinator");
		unsetenv("SPEL_MIRROR_WAIT");
		remove(x);
		/* a mirror whose model is not the one its block names */
		snprintf(y, sizeof y, "%s/tip.mlf2", MI);
		{ FILE *f = fopen(y, "r+b"); if (f) { fseek(f, -9, SEEK_END); int c = fgetc(f); fseek(f, -1, SEEK_CUR); fputc(c ^ 1, f); fclose(f); } }
		fprintf(stderr, "(the refused model below is expected)\n");
		ck(net_sync(u2, mu, F2, &t, 0) == 0 && t.height == 13 && t.from_url >= 1, "a mirror's wrong model is refused, and the coordinator's taken");
		snprintf(x, sizeof x, "%s/tip.mlf2", S2); snprintf(y, sizeof y, "%s/tip.mlf2", F2);
		ck(same_file(x, y) && ch_verify(F, 0) != 0, "the right model; a copy that skipped old deltas cannot verify, as before");
		/* a relay: a farm's devices talk to it alone */
		char R[300], D[300], u3[64];
		snprintf(R, sizeof R, "build/test_net-%ld-relay", pid);
		snprintf(D, sizeof D, "build/test_net-%ld-device", pid);
		snprintf(u3, sizeof u3, "http://127.0.0.1:%d", port + 2);
		int rr[2];
		if (pipe(rr) != 0) return 1;
		fflush(stdout);
		pid_t rl = fork();
		if (rl == 0) { close(rr[0]); _exit(net_relay(u2, R, NULL, port + 2, rr[1]) ? 1 : 0); }
		close(rr[1]);
		ck(rl > 0 && read(rr[0], &rb, 1) == 1, "a relay comes up with a copy of the chain");
		snprintf(x, sizeof x, "%s/tip", S2); snprintf(y, sizeof y, "%s/tip", R);
		ck(same_file(x, y), "its copy is at the coordinator's tip");
		ck(net_sync(u3, "", D, &t, 0) == 0 && t.height == 13, "a device syncs from the relay alone");
		snprintf(y, sizeof y, "%s/tip", D);
		ck(same_file(x, y) && t.round == 3600 && t.seal > t.now, "to the coordinator's tip, on the coordinator's clock");
		ck(ch_train(D, "eve", KEY("eve"), hx) == 0 && net_submit(u3, D, hx) == 0 && count_dlt(S2) == 1, "its delta goes through the relay to the coordinator");
		snprintf(y, sizeof y, "%s/trainers/eve.opt", D); remove(y);
		snprintf(y, sizeof y, "%s/trainers/evie.opt", D); remove(y);
		fprintf(stderr, "(the refusal below is expected)\n");
		ck(ch_train(D, "evie", KEY("eve"), hx) == 0 && net_submit(u3, D, hx) != 0 && count_dlt(S2) == 1, "a second from the same key is refused upstream, through the relay");
		snprintf(y, sizeof y, "%s/pending/%s.dlt", D, hx); remove(y);
		ck(ch_seal(S2) == 0, "the coordinator seals block 14");
		int caught = 0;
		for (int i = 0; i < 60 && !caught; i++) { net_tip n; usleep(250000); caught = net_tip_get(u3, D, &n) == 0 && n.height == 14; }
		ck(caught, "the relay catches up on its own");
		ck(net_sync(u3, "", D, &t, 0) == 0 && t.height == 14, "and the device follows it to block 14");
		kill(rl, SIGTERM);
		waitpid(rl, NULL, 0);
		kill(s2, SIGTERM);
		waitpid(s2, NULL, 0);
		char cmd[1400];
		snprintf(cmd, sizeof cmd, "rm -rf '%s' '%s' '%s' '%s' '%s' '%s'", S2, MI, F, F2, R, D);
		if (system(cmd) != 0) fails++;
	}
	/* ── checks: the round closes, trainers check each other, the block comes early ── */
	{
		char MA[300], MB[300], MC[300], u4[64], q[1400], out[400], vfile[400], rfile[400];
		snprintf(MA, sizeof MA, "build/test_net-%ld-kim", pid);
		snprintf(MB, sizeof MB, "build/test_net-%ld-lee", pid);
		snprintf(MC, sizeof MC, "build/test_net-%ld-max", pid);
		snprintf(out, sizeof out, "build/test_net-%ld.out", pid);
		snprintf(vfile, sizeof vfile, "build/test_net-%ld.vrd", pid);
		snprintf(rfile, sizeof rfile, "build/test_net-%ld.round", pid);
		snprintf(u4, sizeof u4, "http://127.0.0.1:%d", port + 3);
		char hk[65], hl[65], hm[65];
		ck(net_sync(u4, "", MA, &t, 0) == 0 && ch_train(MA, "kim", KEY("kim"), hk) == 0 && net_submit(u4, MA, hk) == 0, "checks: kim trains block 1");
		int up = 0;
		for (int i = 0; i < 60 && !up; i++) { net_tip n; usleep(250000); up = net_tip_get(u4, MA, &n) == 0 && n.height == 1; }
		ck(up, "checks: with nobody to check it, block 1 comes on the clock");
		ck(ch_checker(S3, KEY("kim")->pk) && !ch_checker(S3, KEY("lee")->pk), "checks: kim may check now, lee not yet");
		ck(net_sync(u4, "", MA, &t, 0) == 0 && net_sync(u4, "", MB, &t, 0) == 0 && net_sync(u4, "", MC, &t, 0) == 0 && t.height == 1, "checks: three copies at block 1");
		ck(ch_train(MB, "lee", KEY("lee"), hl) == 0 && net_submit(u4, MB, hl) == 0 && ch_train(MA, "kim", KEY("kim"), hk) == 0 && net_submit(u4, MA, hk) == 0,
		   "checks: lee and kim send deltas for block 2");
		int closed = 0;
		struct timespec c0, c1;
		for (int i = 0; i < 80 && !closed; i++) { net_tip n; usleep(250000); closed = net_tip_get(u4, MA, &n) == 0 && n.checking && n.height == 1; }
		clock_gettime(CLOCK_MONOTONIC, &c0);
		ck(closed, "checks: the clock closes the round: /tip says it is being checked");
		fprintf(stderr, "(the refusals below are expected)\n");
		ck(ch_train(MC, "max", KEY("max"), hm) == 0 && net_submit(u4, MC, hm) != 0, "checks: a delta sent while the round is checked is refused");
		char kx[65], lx[65];
		sha256_hex(KEY("kim")->pk, kx); sha256_hex(KEY("lee")->pk, lx);
		snprintf(q, sizeof q, "curl -s -o '%s' '%s/checks/%s'", out, u4, kx);
		ck(system(q) == 0 && !strcmp(line_of(out), hl), "checks: kim is handed lee's delta, not her own");
		snprintf(q, sizeof q, "curl -s -o '%s' '%s/checks/%s'", out, u4, lx);
		ck(system(q) == 0 && fsize(out) == 0, "checks: lee, no checker yet, is handed nothing");
		snprintf(q, sizeof q, "'%s/pending/%s.dlt'", u4, hl);
		ck(http(q) == 200, "checks: a closed round's delta can be fetched");
		{	/* verdicts that must be refused: a stranger's, one for a key's own delta, a forged one */
			snprintf(q, sizeof q, "curl -s -o '%s' '%s/round'", rfile, u4);
			char dl[700];
			snprintf(dl, sizeof dl, "%s/pending/%s.dlt", MA, hk);   /* kim still holds a copy? no: net_submit removed it */
			ch_verdict v; unsigned char par[32];
			snprintf(dl, sizeof dl, "build/test_net-%ld.dlt", pid);
			char g2[1400];
			snprintf(g2, sizeof g2, "curl -s -o '%s' '%s/pending/%s.dlt'", dl, u4, hl);
			int got = system(q) == 0 && system(g2) == 0 && ch_check(MC, rfile, dl, par, &v) == 0;
			ck(got, "checks: anyone can work out a closed round's loss");
			const char *says[3] = { "checks: a verdict from a key with no passed job is refused",
			                        "checks: a verdict on a key's own delta is refused",
			                        "checks: a forged verdict is refused" };
			for (int kind = 0; got && kind < 3; kind++) {
				ch_verdict w = v;
				ch_verdict_sign(KEY(kind == 1 ? "lee" : kind == 0 ? "max" : "kim"), par, &w);
				if (kind == 2) w.sig[9] ^= 1;
				FILE *f = fopen(vfile, "wb"); if (f) { fwrite(&w, sizeof w, 1, f); fclose(f); }
				snprintf(q, sizeof q, "-H 'Expect:' --data-binary '@%s' '%s/verdict'", vfile, u4);
				ck(http(q) == 400, says[kind]);
			}
			remove(dl); remove(rfile); remove(vfile);
		}
		ck(net_check(u4, MB, KEY("lee")) == 0, "checks: lee's check sends nothing");
		ck(net_check(u4, MA, KEY("kim")) == 1, "checks: kim checks lee's delta, and her verdict is taken");
		int two = 0;
		for (int i = 0; i < 80 && !two; i++) { net_tip n; usleep(250000); two = net_tip_get(u4, MA, &n) == 0 && n.height == 2 && !n.checking; }
		clock_gettime(CLOCK_MONOTONIC, &c1);
		double took = (double)(c1.tv_sec - c0.tv_sec) + (c1.tv_nsec - c0.tv_nsec) / 1e9;
		ck(two && took < 20, "checks: every delta checked, the block comes before the window ends");
		snprintf(q, sizeof q, "'%s/pending/%s.dlt'", u4, hl);
		ck(http(q) == 404, "checks: a sealed round's deltas are not handed out");
		ck(net_sync(u4, "", MA, &t, 0) == 0 && t.height == 2 && ch_checker(S3, KEY("lee")->pk) == (1), "checks: kim's copy follows, and lee may check from now on");
		kill(s3, SIGTERM);
		waitpid(s3, NULL, 0);
		char cmd[1400];
		snprintf(cmd, sizeof cmd, "rm -rf '%s' '%s' '%s' '%s' '%s'", S3, MA, MB, MC, out);
		if (system(cmd) != 0) fails++;
	}
	{ char cmd[900]; snprintf(cmd, sizeof cmd, "rm -rf 'build/test_net-%ld-coord' 'build/test_net-%ld-trainer' 'build/test_net-%ld.txt' 'build/test_net-%ld.mlf2' 'build/test_net-%ld.mlf2.opt'", pid, pid, pid, pid, pid); if (system(cmd) != 0) fails++; }
	printf(fails ? "test_net: %d FAILED\n" : "test_net: all passed\n", fails);
	return fails != 0;
}
