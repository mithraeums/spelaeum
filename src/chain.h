/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* chain.h — Spelaeum (the cave; the tool is spel): block mining applied to training. The chain is the model: block 0 is a
 * starting model, every block after it a weight delta and its parent's hash, so
 * replaying the chain rebuilds the model bit for bit. Miners train on the data the
 * chain's owner gave it, never their own: the hash picks each miner's shard. A
 * miner commits to its delta by hash; the round's seed, made from every commit,
 * then picks the windows the deltas are checked on, from shards nobody mined that
 * round. A delta counts when it cuts the loss there by the difficulty (proof of
 * loss: checking is forward passes, a small part of the mining). Every delta that
 * passes merges into the round's block; each block records who mined what.
 *
 * A chain is a directory:
 *   chain.cfg        the parameters, the data's hash and the genesis hash
 *   genesis.mlf2     block 0, a model hakm-train wrote
 *   ids.bin          the training tokens, encoded once with the genesis tokenizer
 *   blocks/NNNNNN.blk
 *   pending/<commit>.dlt   mined deltas waiting for the next seal
 *   tip, tip.mlf2    the head (height + hash) and its model, rebuilt by a replay
 *   genesis.opt      optional: the optimizer the genesis model was trained with
 *   miners/<name>.opt      each miner's own optimizer, carried from job to job
 * The optimizer files are not part of the chain: they are each miner's business, and
 * nobody checks them. They matter because a cold optimizer's first steps move every
 * weight by about the full rate, and a short job on a trained model spends itself
 * recovering from that.
 *
 * An expert chain (experts > 0, on a model with experts) deals each job a few of
 * every layer's experts the way it deals the shard, from the hash: the job trains
 * those and nothing else, and its delta carries only them. A unit is a range of
 * the weights: 0 = the trunk, everything before the first expert (routers too),
 * trained in trunk rounds only (every block whose height `trunk` divides: all
 * jobs train it, the block is their mean); 1 + l x E + e = expert (l, e). Deltas and blocks there are
 * version 2: a unit list after the head (and the block's contributions),
 * uint32 nunits then uint32 unit[nunits] in increasing order, then the floats
 * unit by unit; n counts the floats carried. A block's units are the union of
 * its merged miners' units, each unit the mean of the deltas that carry it. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#define CH_VERSION 1
#define CH_UNITS 2           /* delta and block version on an expert chain: a unit list before the floats */
#define CH_MINER 32          /* miner name bytes, NUL included */

typedef struct {
	int shards;              /* the training tokens cut into this many equal shards */
	int steps;               /* optimizer steps in one mining job */
	int batch;               /* sequences of ctx tokens per step */
	float lr;                /* peak rate; warmup over the first tenth, cosine to a tenth */
	int eval_windows;        /* windows of ctx tokens a delta is checked on */
	double difficulty;       /* nats per token block 1 must cut; retargets after */
	int stripe;              /* tokens per stripe: stripe k belongs to shard k % shards, so
	                            every shard samples the whole corpus, not one stretch of it */
	int experts;             /* experts of each layer one job trains, the rest frozen; 0 = dense,
	                            every job trains every weight */
	int trunk;               /* on an expert chain: every block whose height this divides is a trunk
	                            round, every job training the trunk and no expert; 0 = never */
} ch_params;

typedef struct {
	ch_params p;
	unsigned char data[32], genesis[32];
} ch_config;

/* One mined delta as it sits in pending/. The commit is sha256 of the whole file. */
typedef struct {
	char magic[4]; uint32_t version;
	unsigned char parent[32];
	uint64_t n;              /* floats in the delta: the model's parameter count, or its units' */
	uint32_t shard, steps;
	char miner[CH_MINER];
	double train_loss;       /* the job's last-step loss, for the log only */
} ch_dhead;

typedef struct {
	char magic[4]; uint32_t version;
	uint64_t height;
	unsigned char parent[32], seed[32];
	double difficulty, parent_loss, new_loss;
	uint32_t ncontrib, nmerged;
	uint64_t n;              /* floats in the delta: 0 when nothing passed */
} ch_bhead;

enum { CH_REJECTED = 0, CH_PASSED = 1, CH_MERGED = 2 };
typedef struct {
	char miner[CH_MINER];
	unsigned char commit[32];
	uint32_t shard, state;
	double loss;             /* on the round's windows, nats per token */
} ch_contrib;

/* Make DIR from a model hakm-train wrote and the text to train on. opt, when not
 * NULL, is that model's MODEL.mlf2.opt: miners start warm from it. */
int ch_init(const char *dir, const char *model, const char *opt, const char *data, const ch_params *p);
int ch_load_config(const char *dir, ch_config *c);
/* The head: height and hash. 0 on success. */
int ch_tip(const char *dir, uint64_t *height, unsigned char hash[32]);

/* The shard a miner works on above a parent. */
uint32_t ch_shard(const unsigned char parent[32], const char *miner, int shards);
/* The experts a miner trains above a parent on an expert chain: per of each of the
 * layers' experts, as units 1 + l x experts + e, sorted. Writes layers x per units
 * and returns that count; -1 unless 1 <= per <= experts. */
int ch_units(const unsigned char parent[32], const char *miner, int layers, int experts, int per, uint32_t *out);
/* Where a window of T+1 tokens starts in shard s, for a random r: a stripe the
 * shard owns, and a place inside it the window fits. */
long ch_window(long ntok, int shards, int stripe, int s, uint64_t r, int T);
/* The round's seed: sha256 of the parent and the commits, sorted. */
void ch_seed(const unsigned char parent[32], const unsigned char (*commits)[32], int n, unsigned char out[32]);
/* Difficulty for the block after one that had ncontrib deltas, npassed of them good. */
double ch_retarget(double d, int ncontrib, int npassed);

/* Mine one job on the tip; writes pending/<commit>.dlt and its hex to commit_hex. */
int ch_mine(const char *dir, const char *miner, char commit_hex[65]);
/* Close the round: check every pending delta, write the block, move the tip.
 * Prints what happened. 0 on success, 1 when nothing was pending. */
int ch_seal(const char *dir);
/* Replay from genesis: links, hashes, seeds, difficulty, states, and the tip's
 * weights bit for bit. full also re-runs each block's loss check. 0 when it holds. */
int ch_verify(const char *dir, int full);
