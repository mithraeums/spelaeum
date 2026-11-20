/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* net.h — a chain over HTTP. One coordinator keeps the chain directory: it hands
 * out the config, the genesis model, the tokens and the blocks as plain files,
 * takes in trained deltas, and seals a round on a clock. Trainers keep a copy of the
 * directory, built from the blocks by replaying them, so a trainer checks every
 * link, hash, seed and difficulty of what it was given before it trains on it.
 *
 * The coordinator is trusted to keep the clock and to keep pending deltas to
 * itself until the seal (that hiding is what stops a trainer picking its delta by
 * the seed it would make); it is not trusted for the chain, which anyone replays.
 * It speaks plain HTTP: put TLS in front of it.
 *
 *   GET  /tip            height, hash, when the round seals, the time now (text)
 *   GET  /status.json    the chain and its last blocks, for a page to show
 *   GET  /chain.cfg /genesis.mlf2 /genesis.opt /ids.bin /tip.mlf2 /blocks/NNNNNN.blk
 *   GET  /heads/N        head parts of blocks N up, end to end (NET_HEADS at most)
 *   POST /submit         a delta; answered with its hash, or 400 and why
 *   GET  /round          the closed round being checked (chain.h, ch_close)
 *   GET  /checks/KEY     the round's deltas this key should check, a commit a line
 *   GET  /pending/C.dlt  a delta of the closed round (never before it closes)
 *   POST /verdict        a ch_verdict from a checker
 *
 * With checks (net_serve_checked), the clock closes the round instead of sealing it:
 * no delta is taken while its deltas are checked, by the trainers themselves, each
 * verdict signed, and the block comes when every delta has its verdicts (or none
 * can get them) or the check window ends. /tip says "phase checking" meanwhile.
 *
 * A mirror is the chain directory on any static host (Hugging Face, R2, a farm's
 * own server, file:// for a shared disk), as net_mirror lays it out: chain.cfg,
 * genesis.mlf2, genesis.opt, ids.bin, blocks/NNN/NNNNNN.blk (a folder a thousand
 * blocks, under Hugging Face's ten thousand a folder), and tip + tip.mlf2, a
 * snapshot that may trail the blocks. The coordinator names it in /tip (files URL,
 * and mirrored H when DIR/mirrored holds the height it has reached); a trainer takes
 * the big files from it first, the coordinator's only when the mirror lacks one.
 * Nothing from a mirror is trusted: every block is linked to the coordinator's
 * heads and every model held to a head's hash, as anything fetched is. So the
 * coordinator sends the small things (the tip, heads, the config) and takes the
 * deltas, and the bytes everyone downloads come from somewhere that serves them free.
 */
#pragma once
#include <stdint.h>
#include <stdio.h>

#define NET_PORT 7717
#define NET_CONNS 64         /* connections served at once; more get a 503 */
#define NET_PENDING 512      /* deltas one round takes */
#define NET_HEADS 256        /* head parts one /heads answer carries */
#define NET_WHOLE 8          /* a copy this many blocks behind or less fetches them whole; further, heads and the model */

/* Serve dir until killed: every round_s seconds, seal. bind_addr NULL = 127.0.0.1.
 * files, when not NULL: the mirror to name in /tip (http:// or https://). after,
 * when not NULL: a program run as `after DIR HEIGHT` once a block is sealed (one at
 * a time; a seal while it still runs is caught up by the next), the place to lay
 * out and upload a mirror. ready_fd >= 0 gets one byte once the port is listening. */
int net_serve(const char *dir, const char *bind_addr, int port, int round_s, const char *files, const char *after, int ready_fd);
/* need: verdicts that must agree to stand for a delta's loss (0: the coordinator
 * works out every one, net_serve); spot: one in this many worked out anyway (0
 * never); window: seconds a closed round waits for its checks (0: NET_CHECK_WINDOW) */
typedef struct { int need, spot, window; } net_checks;
int net_serve_checked(const char *dir, const char *bind_addr, int port, int round_s, const char *files, const char *after,
                      const net_checks *cx, int ready_fd);

#define NET_CHECKS 4         /* deltas one /checks answer hands a checker */
#define NET_CHECK_WINDOW 60  /* seconds a closed round waits for its checks */
#define NET_RELAY_POLL 5     /* seconds between a relay's looks at its coordinator */
#define NET_MIRROR_WAIT 90   /* seconds a trainer waits for the mirror to reach the tip before taking it from the
                                coordinator; SPEL_MIRROR_WAIT overrides */

/* Be a relay for the coordinator at url until killed: keep dir a checked copy of its
 * chain (as net_sync keeps a trainer's), and answer as it would (/tip, /heads, the
 * files, /status.json) from that copy, passing each delta on and its answer back.
 * For a farm: its devices talk to the relay over the farm's own network, the
 * coordinator and the mirror see one client. ready_fd as net_serve's. */
int net_relay(const char *url, const char *dir, const char *bind_addr, int port, int ready_fd);

typedef struct {
	uint64_t height; char hash[65]; long long seal, now; int round;
	char files[520];         /* the mirror the coordinator names, "" for none */
	uint64_t mirrored;       /* how far the mirror has got, when has_mirrored */
	int has_mirrored;
	int from_files, from_url;   /* net_sync: big files (data, genesis, blocks, models) from the mirror, from the coordinator */
	int checking;            /* the round is closed and its deltas being checked */
} net_tip;
int net_tip_get(const char *url, const char *dir, net_tip *t);
/* Bring dir up to the coordinator's tip and check it is the one the coordinator
 * names. A few blocks behind: fetch them whole and apply them. New, or further
 * behind: fetch the heads and the model whole, and take the model when its hash is
 * the last head's (old deltas never travel), and keep only the last NET_WHOLE
 * blocks whole. whole: fetch every block whole and keep them, for a verifier.
 * files: a mirror (http://, https:// or file://) in place of the one the
 * coordinator names, which may only be http:// or https://; "" for none at all.
 * A copy far behind joins at the mirror's snapshot when it lies between, then
 * takes the blocks after it whole. Makes dir when it is not there. */
int net_sync(const char *url, const char *files, const char *dir, net_tip *t, int whole);
/* Check what the coordinator's closed round hands this key, on dir (the round's
 * parent), each loss signed and sent as a verdict. How many it sent, or -1. */
#include "chain.h"
int net_check(const char *url, const char *dir, const ch_key *key);
/* Send pending/<hex>.dlt; removed here once the coordinator has it. */
int net_submit(const char *url, const char *dir, const char *hex);
/* Lay out (or bring up to date) the mirror of dir in out: every file it lacks, and
 * a new snapshot once the tip is every blocks past the one out has. Each path it
 * writes goes to added (NULL for none), one a line, for an uploader. Copies, never
 * links: a block pruned later must stay whole in its mirror. */
int net_mirror(const char *dir, const char *out, uint64_t every, FILE *added);
