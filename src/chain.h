/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* chain.h — Spelaeum (the cave; the tool is spel): many machines training one model, a block at a time. The chain is the model: block 0 is a
 * starting model, every block after it a weight delta and its parent's hash, so
 * replaying the chain rebuilds the model bit for bit. Trainers train on the data the
 * chain's owner gave it, never their own: the hash picks each trainer's shard. A
 * trainer commits to its delta by hash; the round's seed, made from every commit,
 * then picks the windows the deltas are checked on, from shards nobody trained that
 * round. A delta counts when it cuts the loss there by the difficulty (proof of
 * loss: checking is forward passes, a small part of the training, worked out in
 * integers by the engine's iloss.h, so every machine gets the same loss to the
 * bit and a check is a match, not a tolerance). Every delta that
 * passes merges into the round's block; each block records who trained what.
 *
 * A chain is a directory:
 *   chain.cfg        the parameters, the data's hash and the genesis hash
 *   genesis.mlf2     block 0, a model hakm-train wrote
 *   ids.bin          the training tokens, encoded once with the genesis tokenizer; the
 *                    data's hash is over its stripes' hashes, so one stripe checks alone
 *   blocks/NNNNNN.blk      a head part (the head, the contributions, an expert chain's
 *                          units), then the floats; the block's hash is the head part's,
 *                          and the head names the floats' hash (body) and the hash of
 *                          the model they make (model, sha256 of every weight as a
 *                          float). So heads alone link the chain, a block whose floats
 *                          were cut off (pruned) still links, and a copy can start
 *                          from heads and a model fetched whole
 *   pending/<commit>.dlt   trained deltas waiting for the next seal
 *   tip, tip.mlf2    the head (height + hash) and its model, rebuilt by a replay
 *   genesis.opt      optional: the optimizer the genesis model was trained with
 *   trainers/<name>.opt      each trainer's own optimizer, carried from job to job
 * A trainer is its key: an Ed25519 public key deals its shard and its experts, and
 * every delta it sends is signed with it, so nobody trains in its name. The name
 * beside the key is a label for the credit board, nothing more: two keys can
 * share one. The signature trails the delta, 64 bytes after the floats; the
 * commit is sha256 of the file before it, and the key signs "spelaeum delta\n",
 * the parent, the commit and the name's CH_NAME bytes. A block keeps each
 * contribution's key and signature, so a replay checks that every trainer it
 * names sent that delta under that name.
 * The optimizer files are not part of the chain: they are each trainer's business, and
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
 * its merged trainers' units, each unit the mean of the deltas that carry it. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#define CH_VERSION 1
#define CH_UNITS 2           /* delta and block version on an expert chain: a unit list before the floats */
#define CH_NAME 32          /* trainer name bytes, NUL included */
#define CH_SIG 64            /* the signature that trails a delta */

typedef struct {
	int shards;              /* the training tokens cut into this many equal shards */
	int steps;               /* optimizer steps in one training job */
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

/* One trained delta as it sits in pending/. The commit is sha256 of the whole file. */
typedef struct {
	char magic[4]; uint32_t version;
	unsigned char parent[32];
	uint64_t n;              /* floats in the delta: the model's parameter count, or its units' */
	uint32_t shard, steps;
	char trainer[CH_NAME];
	double train_loss;       /* the job's last-step loss, for the log only */
	unsigned char key[32];   /* the trainer: what deals its job and signs its delta */
} ch_dhead;

typedef struct {
	char magic[4]; uint32_t version;
	uint64_t height;
	unsigned char parent[32], seed[32];
	double difficulty, parent_loss, new_loss;
	uint32_t ncontrib, nmerged;
	uint64_t n;              /* floats in the delta: 0 when nothing passed */
	unsigned char body[32];  /* sha256 of the floats */
	unsigned char model[32]; /* sha256 of the model's weights once they are applied */
} ch_bhead;

enum { CH_REJECTED = 0, CH_PASSED = 1, CH_MERGED = 2 };
typedef struct {
	char trainer[CH_NAME];
	unsigned char commit[32];
	uint32_t shard, state;
	double loss;             /* on the round's windows, nats per token, from the integer pass */
	unsigned char key[32], sig[CH_SIG];
} ch_contrib;

/* A trainer's key: the secret seed and the public key it makes. */
typedef struct { unsigned char seed[32], pk[32]; } ch_key;
void ch_key_from(const unsigned char seed[32], ch_key *k);
/* The key kept in path ("spelaeum key", then the seed in hex). make: when there is
 * none, make one from /dev/urandom, mode 0600, its directory too. 0 on success. */
int ch_key_load(const char *path, ch_key *k, int make);

/* Make DIR from a model hakm-train wrote and the text to train on. opt, when not
 * NULL, is that model's MODEL.mlf2.opt: trainers start warm from it. */
int ch_init(const char *dir, const char *model, const char *opt, const char *data, const ch_params *p);
int ch_load_config(const char *dir, ch_config *c);
/* The head: height and hash. 0 on success. */
int ch_tip(const char *dir, uint64_t *height, unsigned char hash[32]);

/* The shard a trainer (its public key) works on above a parent. */
uint32_t ch_shard(const unsigned char parent[32], const unsigned char key[32], int shards);
/* The experts a trainer trains above a parent on an expert chain: per of each of the
 * layers' experts, as units 1 + l x experts + e, sorted. Writes layers x per units
 * and returns that count; -1 unless 1 <= per <= experts. */
int ch_units(const unsigned char parent[32], const unsigned char key[32], int layers, int experts, int per, uint32_t *out);
/* Where a window of T+1 tokens starts in shard s, for a random r: a stripe the
 * shard owns, and a place inside it the window fits. */
int64_t ch_window(int64_t ntok, int shards, int stripe, int s, uint64_t r, int T);
/* The round's seed: sha256 of the parent and the commits, sorted. */
void ch_seed(const unsigned char parent[32], const unsigned char (*commits)[32], int n, unsigned char out[32]);
/* Difficulty for the block after one that had ncontrib deltas, npassed of them good. */
double ch_retarget(double d, int ncontrib, int npassed);

/* Train one job on the tip as key, labelled trainer; writes pending/<commit>.dlt, signed,
 * and its hex to commit_hex. */
int ch_train(const char *dir, const char *trainer, const ch_key *key, char commit_hex[65]);
/* Close the round: check every pending delta, write the block, move the tip.
 * Prints what happened. 0 on success, 1 when nothing was pending. */
int ch_seal(const char *dir);
/* Replay from genesis: links, hashes, seeds, difficulty, states, and the tip's
 * weights bit for bit. full also re-runs each block's loss check. 0 when it holds. */
int ch_verify(const char *dir, int full);
/* A trainer's copy brought up to the blocks it holds, from the tip it has: each new
 * block checked as verify checks it, its delta applied, the model's hash held to
 * the last head's; then tip and tip.mlf2 written. Every new block must be whole. */
int ch_advance(const char *dir);
/* A trainer's copy from heads alone: every head above the tip it has (or genesis, with
 * the data and genesis checked) checked as verify checks it, no delta needed; then
 * DIR/model, a model fetched whole, becomes tip.mlf2 when its hash is the last head's. */
int ch_join(const char *dir, const char *model);
/* The head part of block h, malloc'd: what its hash covers. */
int ch_head_part(const char *dir, uint64_t h, unsigned char **out, size_t *len);
/* Head parts end to end (a coordinator's answer from block `from` up) into DIR's
 * blocks as pruned files, where it has none, none past upto (UINT64_MAX for all).
 * *got = how many. */
int ch_put_heads(const char *dir, const unsigned char *buf, size_t len, uint64_t from, uint64_t upto, uint64_t *got);
/* Cut the floats off every block more than keep below the tip: the heads stay and
 * the chain still links; verify needs them back, a trainer never does. Prints how
 * much it freed unless quiet. */
int ch_prune(const char *dir, uint64_t keep, int quiet);
/* A new copy's tip at genesis: its data and genesis checked, tip.mlf2 = genesis.mlf2.
 * Nothing done when it has a tip. */
int ch_start(const char *dir);
/* Take a delta file a trainer sent: checked against the tip like the seal checks one,
 * one a trainer a round, then moved into pending/ under its hash. 0 and its hex, or
 * -1 and why. The file is left where it was when refused. */
int ch_submit(const char *dir, const char *file, char hex[65], const char **why);
/* The chain as JSON: height, tip, the next round, the last blocks (all when last
 * <= 0) newest first with each one's trainers. */
#include <stdio.h>
int ch_status_json(const char *dir, FILE *out, int last);

/* Checks: a coordinator closes the round (ch_close), trainers check each other's
 * deltas (ch_check) and sign what they found (a verdict), and the seal takes a
 * delta's loss from `need` verdicts that agree rather than working it out, but for
 * one in every `spot`, picked by the seed, that it works out anyway: checkers who
 * agreed on a wrong loss are barred. A checker is a key with a passed job on this
 * chain (DIR/checkers, kept by the seal), not barred (DIR/barred), never a delta's
 * own. The verdicts are the coordinator's business, not the chain's: a block holds
 * the losses, and verify --full re-runs the block's. */
typedef struct {
	unsigned char key[32], commit[32];
	double loss;
	unsigned char sig[CH_SIG];
} ch_verdict;                /* 136 bytes, as sent and as kept in pending/<commit>.vrd */
/* Close the round: the pending deltas checked (the bad, stale and second set aside,
 * as the seal does) and DIR/round written: "spelaeum round", height, parent, seed,
 * then "delta COMMIT KEY SHARD" a line, by commit. 0; 1 when nothing is pending. */
int ch_close(const char *dir);
/* Check one delta of a round on DIR's tip, its parent: the round's seed and shards
 * worked out here, the delta checked as the seal checks it, then its loss on the
 * round's windows into v (commit and loss; sign it with ch_verdict_sign). */
int ch_check(const char *dir, const char *round, const char *delta, unsigned char parent[32], ch_verdict *v);
void ch_verdict_sign(const ch_key *k, const unsigned char parent[32], ch_verdict *v);
int ch_verdict_ok(const unsigned char parent[32], const ch_verdict *v);
/* May this key check on DIR's chain: a checker, not barred. */
int ch_checker(const char *dir, const unsigned char key[32]);
/* ch_seal, taking losses from verdicts as above; *taken = how many it took. */
int ch_seal_checked(const char *dir, int need, int spot, int *taken);

/* Credit: what a key's work on the chain comes to, read from the heads alone (a
 * pruned copy has it all), so anyone with a copy that verify, sync or join checked
 * counts the same. A job counts when its delta passed its check, merged or not:
 * a refused one did no work the chain can see. Each passed job is worth the
 * operations it took, estimated the same way everywhere: 6 x the weights a token
 * passes through (every weight on a dense model; the trunk and top_k experts of
 * each layer on one with experts) x the tokens it trained on (steps x batch x ctx). */
typedef struct {
	unsigned char key[32];
	char trainer[CH_NAME];    /* the label on its latest delta */
	uint64_t passed, merged, refused;
	double ops;              /* passed x the operations of one job */
} ch_credit;
/* Every key with a delta on the chain, the most operations first, into *out
 * (malloc'd); *job = the operations of one job. The count, or -1. */
long ch_credits(const char *dir, ch_credit **out, double *job);
/* The grades: Corax, Nymphus, Miles, Leo, Perses, Heliodromus, Pater, the first at
 * CH_GRADE_FIRST operations and each ten times the last. Returns 0 (none yet) to 7;
 * *name the grade's, NULL for none; *next the operations the next one needs, 0
 * past Pater. */
#define CH_GRADE_FIRST 1e15
#define CH_GRADES 7
int ch_grade(double ops, const char **name, double *next);
