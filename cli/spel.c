/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/*
 * spel — Spelaeum: block mining applied to training. The chain is the model.
 *
 *   spel init   DIR --model MODEL.mlf2 --data TEXT [--opt MODEL.mlf2.opt] [options]
 *   spel mine   DIR --miner NAME
 *   spel seal   DIR
 *   spel verify DIR [--full]
 *   spel status DIR
 *
 * DIR/tip.mlf2 is the chain's model, and hakm runs it as is. See src/chain.h. Built on the hako engine's trainer (../hako).
 */
#include "../src/chain.h"
#include "../src/sha256.h"

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
		"       spel mine   DIR --miner NAME     train on the shard the tip gives NAME\n"
		"       spel seal   DIR             check every pending delta, write the next block\n"
		"       spel verify DIR [--full]    replay from genesis (--full re-runs every loss check)\n"
		"       spel status DIR\n"
		"every command: --threads N (every core)  --nice N (19, the lowest)\n");
}

int main(int argc, char **argv) {
	if (argc < 3) { usage(); return 2; }
	const char *cmd = argv[1], *dir = argv[2], *model = NULL, *opt = NULL, *data = NULL, *miner = NULL;
	ch_params p = { 64, 200, 8, 3e-4f, 32, 0.002, 4096, 0, 0 };
	int full = 0, niceness = 19;
	for (int i = 3; i < argc; i++) {
		const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
		#define OPT(name) (!strcmp(a, name) && v && ++i)
		if      (OPT("--model")) model = v;
		else if (OPT("--opt")) opt = v;
		else if (OPT("--data")) data = v;
		else if (OPT("--miner")) miner = v;
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
		else if (OPT("--nice")) niceness = atoi(v);
		else if (!strcmp(a, "--full")) full = 1;
		else { usage(); return 2; }
		#undef OPT
	}
	if (niceness > 0 && setpriority(PRIO_PROCESS, 0, niceness) != 0) perror("chain: setpriority");

	if (!strcmp(cmd, "init")) {
		if (!model || !data) { usage(); return 2; }
		return ch_init(dir, model, opt, data, &p) ? 1 : 0;
	}
	if (!strcmp(cmd, "mine")) {
		if (!miner) { usage(); return 2; }
		char hex[65];
		if (ch_mine(dir, miner, hex)) return 1;
		printf("%s\n", hex);
		return 0;
	}
	if (!strcmp(cmd, "seal")) return ch_seal(dir) < 0 ? 1 : 0;
	if (!strcmp(cmd, "verify")) return ch_verify(dir, full) ? 1 : 0;
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
