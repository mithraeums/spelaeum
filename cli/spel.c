/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/*
 * spel — Spelaeum: many machines training one model, a block at a time. The chain is the model.
 *
 *   spel init   DIR --model MODEL.mlf2 --data TEXT [--opt MODEL.mlf2.opt] [options]
 *   spel train  DIR --name NAME
 *   spel seal   DIR [--checks K [--spot N]]
 *   spel close  DIR                      close the round for checks: DIR/round
 *   spel verify DIR [--full]
 *   spel status DIR
 *   spel serve  DIR [--files URL] [--after PROGRAM]   the coordinator (src/net.h)
 *   spel sync   DIR --url URL [--files URL]
 *   spel train  DIR --name NAME --url URL [--files URL] [--loop]
 *   spel mirror DIR OUT [--every N]      the chain laid out for a static host
 *   spel relay  DIR --url URL            a farm's own coordinator: a checked copy, deltas passed on
 *   spel prune  DIR [--keep N]           cut the floats off old blocks, heads kept
 *   spel key    [--key FILE]             this machine's trainer key: made on first use
 *   spel credit DIR [KEY]                what each key's passed jobs come to, and its grade
 *
 * DIR/tip.mlf2 is the chain's model, and hakm runs it as is. See src/chain.h. Built on the hako engine's trainer (../hako).
 */
#include "../src/chain.h"
#include "../src/sha256.h"
#include "../src/net.h"
#include "train.h"
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

static void usage(void) {
	fprintf(stderr,
		"usage: spel init   DIR --model MODEL.mlf2 --data TEXT [--opt MODEL.mlf2.opt]\n"
		"                         [--shards N (64)] [--stripe N tokens (4096)] [--steps N (200)]\n"
		"                         [--batch N (8)] [--lr F (3e-4)] [--eval N (32)] [--difficulty F (0.002)]\n"
		"                         [--experts N (0)]  each job trains N of every layer's experts, the rest\n"
		"                                            frozen (a model with experts; 0 trains every weight)\n"
		"                         [--trunk N (0)]    with --experts: every Nth block trains the trunk\n"
		"                                            (attention, routers) instead, experts frozen\n"
		"       spel train  DIR --name NAME      train on the shard the tip gives this key; NAME labels it\n"
		"                         [--key FILE]       the key (~/.spel/key, made when there is none)\n"
		"                         [--url URL]        train for a coordinator: DIR is this machine's copy of\n"
		"                                            its chain (made and brought up to date here), the\n"
		"                                            delta is sent there\n"
		"                         [--loop]           with --url: round after round until stopped\n"
		"                         [--files URL]      with --url: the mirror to take big files from, in place\n"
		"                                            of the one the coordinator names (file:// too; \"\" for none)\n"
		"                         [--cpu]            train on the CPU though spel was built for the GPU (make GPU=1)\n"
		"       spel sync   DIR --url URL        fetch a coordinator's chain into DIR whole and replay it (to\n"
		"                                        verify it; a trainer's copy keeps only the last few blocks whole)\n"
		"       spel serve  DIR [--port N (7717)] [--bind ADDR (127.0.0.1)] [--round SECONDS (600)]\n"
		"                                        be the coordinator: hand out the chain, take deltas,\n"
		"                                        seal on the clock. plain HTTP: put TLS in front\n"
		"                         [--files URL]      name a mirror in /tip: trainers take big files from it\n"
		"                         [--after PROGRAM]  run PROGRAM DIR HEIGHT after each block (to mirror it)\n"
		"                         [--checks K]       close each round and let trainers check it: K verdicts that\n"
		"                                            agree stand for a delta's loss (0, the default: work out all)\n"
		"                         [--spot N (8)]     with --checks: work out one in N anyway, barring liars\n"
		"                         [--check-window S (60)]  with --checks: seconds a closed round waits\n"
		"       spel relay  DIR --url URL [--port N (7717)] [--bind ADDR (127.0.0.1)]\n"
		"                                        for a farm: keep DIR a checked copy of URL's chain and answer\n"
		"                                        its devices as the coordinator would, deltas passed on\n"
		"       spel mirror DIR OUT [--every N (8)]  lay out DIR for a static host in OUT (blocks in folders of\n"
		"                                        a thousand), a new snapshot every N blocks; prints each path\n"
		"                                        it writes, for an uploader\n"
		"       spel seal   DIR             check every pending delta, write the next block\n"
		"                         [--checks K] [--spot N]  taking losses from K agreeing verdicts\n"
		"       spel close  DIR             close the round (DIR/round) for trainers to check\n"
		"       spel verify DIR [--full]    replay from genesis (--full re-runs every loss check)\n"
		"       spel status DIR\n"
		"       spel prune  DIR [--keep N (64)]  cut the floats off every block but the last N: heads stay,\n"
		"                                        the chain still links and trainers still join; verify needs them\n"
		"       spel key    [--key FILE]    print this machine's public key, making the key first if need be\n"
		"                                   (~/.spel/key is a text file: copy it to keep your credit, or to\n"
		"                                   train as the same key on another machine)\n"
		"       spel credit DIR [KEY]       every key's passed jobs, their operations and grade, Corax to Pater;\n"
		"                                   KEY (as spel key prints it) for one, and how far to the next grade\n"
		"every command: --threads N (every core)  --nice N (19, the lowest)\n");
}

/* --key, else ~/.spel/key */
static int key_path(const char *given, char *out, size_t cap) {
	const char *home = getenv("HOME");
	int k = given ? snprintf(out, cap, "%s", given) : home && *home ? snprintf(out, cap, "%s/.spel/key", home) : -1;
	if (k < 0 || (size_t)k >= cap) { fprintf(stderr, "spel: no key path: give --key FILE\n"); return -1; }
	return 0;
}
static int load_key(const char *given, ch_key *k) {
	char path[4096];
	if (key_path(given, path, sizeof path)) return -1;
	int had = access(path, F_OK) == 0;
	if (ch_key_load(path, k, 1)) return -1;
	char hx[65];
	sha256_hex(k->pk, hx);   /* hex, nothing hashed */
	if (!had) fprintf(stderr, "spel: made a key in %s, public %s. keep it: it is who trained\n", path, hx);
	return 0;
}

int main(int argc, char **argv) {
	if (argc >= 2 && !strcmp(argv[1], "key")) {
		const char *given = argc == 4 && !strcmp(argv[2], "--key") ? argv[3] : NULL;
		if (argc != 2 && !given) { usage(); return 2; }
		ch_key k;
		if (load_key(given, &k)) return 1;
		char hx[65];
		sha256_hex(k.pk, hx);
		printf("%s\n", hx);
		return 0;
	}
	if (argc >= 3 && !strcmp(argv[1], "credit")) {
		unsigned char want[32];
		int one = argc == 4;
		for (int i = 0; one && i < 32; i++) {
			unsigned v;
			if (strlen(argv[3]) != 64 || sscanf(argv[3] + 2 * i, "%2x", &v) != 1) one = -1; else want[i] = (unsigned char)v;
		}
		if (argc > 4 || one < 0) { usage(); return 2; }
		ch_credit *cr = NULL;
		double job = 0, next;
		long n = ch_credits(argv[2], &cr, &job);
		if (n < 0) return 1;
		const char *g;
		char kx[65];
		if (one) {
			ch_credit z = { .passed = 0 }, *e = &z;
			for (long i = 0; i < n; i++) if (!memcmp(cr[i].key, want, 32)) e = &cr[i];
			int gi = ch_grade(e->ops, &g, &next);
			printf("key        %s\n", argv[3]);
			if (e->trainer[0]) printf("name       %s\n", e->trainer);
			printf("passed     %llu jobs (%llu merged), %llu refused\n", (unsigned long long)e->passed, (unsigned long long)e->merged, (unsigned long long)e->refused);
			printf("operations %.3g\n", e->ops);
			if (!gi) printf("grade      none yet: Corax at %.0e, %.1f%% there\n", next, 100 * e->ops / next);
			else if (next > 0) {
				const char *ng;
				ch_grade(next, &ng, NULL);
				printf("grade      %s; next %s at %.0e, %.1f%% there\n", g, ng, next, 100 * e->ops / next);
			}
			else printf("grade      %s, the last\n", g);
		} else {
			printf("one passed job = %.3g operations; grades from %.0e, each ten times the last\n", job, CH_GRADE_FIRST);
			printf("%-12s %10s %7s %7s %8s  %-16s %s\n", "grade", "operations", "passed", "merged", "refused", "name", "key");
			for (long i = 0; i < n; i++) {
				sha256_hex(cr[i].key, kx);   /* hex, nothing hashed */
				ch_grade(cr[i].ops, &g, NULL);
				printf("%-12s %10.3g %7llu %7llu %8llu  %-16s %.16s\n", g ? g : "-", cr[i].ops, (unsigned long long)cr[i].passed,
				       (unsigned long long)cr[i].merged, (unsigned long long)cr[i].refused, cr[i].trainer[0] ? cr[i].trainer : "-", kx);
			}
		}
		free(cr);
		return 0;
	}
	if (argc < 3) { usage(); return 2; }
	const char *cmd = argv[1], *dir = argv[2], *model = NULL, *opt = NULL, *data = NULL, *trainer = NULL, *url = NULL, *bind_addr = NULL, *keyf = NULL;
	const char *files = NULL, *after = NULL, *out = NULL;
	long every = NET_WHOLE;
	int need = 0, spot = 8, window = NET_CHECK_WINDOW;
	int first = 3;
	if (!strcmp(cmd, "mirror")) { if (argc < 4) { usage(); return 2; } out = argv[3]; first = 4; }
	ch_params p = { 64, 200, 8, 3e-4f, 32, 0.002, 4096, 0, 0 };
	int full = 0, niceness = 19, loop = 0, port = NET_PORT, round_s = 600;
	long keep = 64;
	for (int i = first; i < argc; i++) {
		const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
		#define OPT(name) (!strcmp(a, name) && v && ++i)
		if      (OPT("--model")) model = v;
		else if (OPT("--opt")) opt = v;
		else if (OPT("--data")) data = v;
		else if (OPT("--name")) trainer = v;
		else if (OPT("--key")) keyf = v;
		else if (OPT("--url")) url = v;
		else if (OPT("--files")) files = v;
		else if (OPT("--after")) after = v;
		else if (OPT("--every")) every = atol(v);
		else if (OPT("--checks")) need = atoi(v);
		else if (OPT("--spot")) spot = atoi(v);
		else if (OPT("--check-window")) window = atoi(v);
		else if (OPT("--bind")) bind_addr = v;
		else if (OPT("--port")) port = atoi(v);
		else if (OPT("--round")) round_s = atoi(v);
		else if (!strcmp(a, "--loop")) loop = 1;
		else if (OPT("--shards")) p.shards = atoi(v);
		else if (OPT("--stripe")) p.stripe = atoi(v);
		else if (OPT("--steps")) p.steps = atoi(v);
		else if (OPT("--batch")) p.batch = atoi(v);
		else if (OPT("--lr")) p.lr = (float)atof(v);
		else if (OPT("--eval")) p.eval_windows = atoi(v);
		else if (OPT("--difficulty")) p.difficulty = atof(v);
		else if (OPT("--experts")) p.experts = atoi(v);
		else if (OPT("--trunk")) p.trunk = atoi(v);
		else if (OPT("--threads")) setenv("HAKM_THREADS", v, 1);
		else if (!strcmp(a, "--cpu")) tr_use_gpu(0);
		else if (OPT("--nice")) niceness = atoi(v);
		else if (!strcmp(a, "--full")) full = 1;
		else if (OPT("--keep")) keep = atol(v);
		else { usage(); return 2; }
		#undef OPT
	}
	if (niceness > 0 && setpriority(PRIO_PROCESS, 0, niceness) != 0) perror("chain: setpriority");

	if (!strcmp(cmd, "init")) {
		if (!model || !data) { usage(); return 2; }
		return ch_init(dir, model, opt, data, &p) ? 1 : 0;
	}
	if (!strcmp(cmd, "train")) {
		if (!trainer) { usage(); return 2; }
		char hex[65];
		ch_key key;
		if (load_key(keyf, &key)) return 1;
		if (!url) {
			if (ch_train(dir, trainer, &key, hex)) return 1;
			printf("%s\n", hex);
			return 0;
		}
		for (;;) {
			net_tip t, now;
			if (net_sync(url, files, dir, &t, 0)) {   /* looping: the coordinator may be down or not up yet */
				if (!loop) return 1;
				fprintf(stderr, "train: could not sync with %s, trying again in 30 s\n", url);
				sleep(30);
				continue;
			}
			if (ch_train(dir, trainer, &key, hex)) return 1;
			if (net_submit(url, dir, hex)) {   /* too late for the round, or refused: the next one may take it */
				fprintf(stderr, "train: the coordinator did not take %.16s\n", hex);
				if (!loop) return 1;
			} else printf("%s\n", hex);
			if (!loop) return 0;
			fflush(stdout);
			int checked = 0;
			do {   /* until the round seals and the tip moves, checking others' deltas once it closes */
				long long wait = t.seal - t.now;
				sleep(wait > 30 ? 30 : wait > 2 ? (unsigned)wait : 2);
				if (net_tip_get(url, dir, &now)) { sleep(10); now = t; now.now = t.seal; continue; }
				t.seal = now.seal; t.now = now.now;
				if (now.checking && now.height == t.height && !checked) {
					int n = net_check(url, dir, &key);
					checked = 1;
					if (n > 0) fprintf(stderr, "train: checked %d of the round's deltas\n", n);
				}
			} while (now.height == t.height);
		}
	}
	if (!strcmp(cmd, "sync")) {
		net_tip t;
		if (!url) { usage(); return 2; }
		if (net_sync(url, files, dir, &t, 1)) return 1;
		printf("height %llu\ntip %s\n", (unsigned long long)t.height, t.hash);
		if (t.from_files + t.from_url) printf("fetched %d files from the mirror, %d from the coordinator\n", t.from_files, t.from_url);
		return 0;
	}
	if (!strcmp(cmd, "serve")) {
		if (need < 0 || spot < 0 || window < 1) { usage(); return 2; }
		net_checks cx = { need, spot, window };
		return net_serve_checked(dir, bind_addr, port, round_s, files, after, &cx, -1) ? 1 : 0;
	}
	if (!strcmp(cmd, "close")) { int r = ch_close(dir); if (r == 1) fprintf(stderr, "close: nothing pending\n"); return r < 0 ? 1 : 0; }
	if (!strcmp(cmd, "relay")) {
		if (!url) { usage(); return 2; }
		return net_relay(url, dir, bind_addr, port, -1) ? 1 : 0;
	}
	if (!strcmp(cmd, "mirror")) {
		if (every < 1) { usage(); return 2; }
		return net_mirror(dir, out, (uint64_t)every, stdout) ? 1 : 0;
	}
	if (!strcmp(cmd, "seal")) {
		if (need <= 0) return ch_seal(dir) < 0 ? 1 : 0;
		int taken = 0, r = ch_seal_checked(dir, need, spot, &taken);
		if (r == 0) fprintf(stderr, "seal: %d deltas taken from trainers' checks\n", taken);
		return r < 0 ? 1 : 0;
	}
	if (!strcmp(cmd, "verify")) return ch_verify(dir, full) ? 1 : 0;
	if (!strcmp(cmd, "prune")) {
		if (keep < 0) { usage(); return 2; }
		return ch_prune(dir, (uint64_t)keep, 0) ? 1 : 0;
	}
	if (!strcmp(cmd, "status")) {
		ch_config c; uint64_t h; unsigned char hash[32];
		if (ch_load_config(dir, &c) || ch_tip(dir, &h, hash)) return 1;
		char hx[65], gx[65];
		sha256_hex(hash, hx); sha256_hex(c.genesis, gx);
		printf("height %llu\ntip %s\ngenesis %s\nshards %d of stripes of %d, %d steps of batch %d, lr %g, %d eval windows, start difficulty %g\n",
		       (unsigned long long)h, hx, gx, c.p.shards, c.p.stripe, c.p.steps, c.p.batch, (double)c.p.lr, c.p.eval_windows, c.p.difficulty);
		if (c.p.experts) printf("experts %d of each layer per job, the rest frozen\n", c.p.experts);
		if (c.p.trunk) printf("trunk   every block whose height %d divides trains the trunk instead\n", c.p.trunk);
		return 0;
	}
	usage();
	return 2;
}
