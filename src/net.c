/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* net.c — a chain over HTTP: the coordinator, and a trainer's side of it (curl as a
 * subprocess, never a shell). See net.h. */
#include "net.h"
#include "chain.h"
#include "sha256.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define PATHN 4096
#define HEADN 8192

static int pj(char *o, const char *dir, const char *name) {
	int k = snprintf(o, PATHN, "%s/%s", dir, name);
	return k < 0 || k >= PATHN ? -1 : 0;
}

/* a URL handed to curl: a scheme this allows, at most 512 bytes, no spaces or quotes */
static int clean_url(const char *url) {
	size_t n = strlen(url);
	if (n > 512) return 0;
	for (size_t i = 0; i < n; i++) if ((unsigned char)url[i] <= ' ' || url[i] == '"' || url[i] == '\'') return 0;
	return 1;
}
static int url_ok(const char *url) { return (!strncmp(url, "http://", 7) || !strncmp(url, "https://", 8)) && clean_url(url); }
/* a mirror: http(s), or file:// when this machine was told it, never when a coordinator names it */
static int files_ok(const char *url, int local) { return (url_ok(url) || (local && !strncmp(url, "file://", 7))) && clean_url(url); }
/* where block h sits in a mirror: a folder a thousand blocks */
static void mirror_block(char *out, size_t cap, uint64_t h) {
	snprintf(out, cap, "blocks/%03llu/%06llu.blk", (unsigned long long)(h / 1000), (unsigned long long)h);
}

/* ── the coordinator ────────────────────────────────────────────────────── */

/* what one server answers as: a coordinator, or a relay (upstream set) that serves
 * its own checked copy of the upstream chain and passes deltas on */
typedef struct { const char *dir, *files, *after, *upstream; int round_s, need, spot, check_s; } srv;

static int send_all(int fd, const void *b, size_t n) {
	const char *p = b;
	while (n) {
		ssize_t w = write(fd, p, n);
		if (w < 0 && errno == EINTR) continue;
		if (w <= 0) return -1;
		p += w; n -= (size_t)w;
	}
	return 0;
}

/* every answer closes the connection and may be read by a page on another origin */
static void answer(int fd, int code, const char *type, const void *body, size_t len) {
	char h[256];
	const char *msg = code == 200 ? "OK" : code == 204 ? "No Content" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" :
	                  code == 405 ? "Method Not Allowed" : code == 413 ? "Payload Too Large" : "Service Unavailable";
	int k = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nAccess-Control-Allow-Origin: *\r\n"
	                 "Access-Control-Allow-Headers: Content-Type\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", code, msg, type, len);
	if (send_all(fd, h, (size_t)k) == 0 && len) send_all(fd, body, len);
}
static void say(int fd, int code, const char *text) { answer(fd, code, "text/plain; charset=utf-8", text, strlen(text)); }

static void send_file(int fd, const char *path, const char *type) {
	int f = open(path, O_RDONLY);
	struct stat st;
	if (f < 0 || fstat(f, &st) != 0 || !S_ISREG(st.st_mode)) { if (f >= 0) close(f); say(fd, 404, "not here\n"); return; }
	char h[256], buf[65536];
	int k = snprintf(h, sizeof h, "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %lld\r\nAccess-Control-Allow-Origin: *\r\n"
	                 "Cache-Control: no-store\r\nConnection: close\r\n\r\n", type, (long long)st.st_size);
	if (send_all(fd, h, (size_t)k) == 0) {
		ssize_t r;
		while ((r = read(f, buf, sizeof buf)) > 0) if (send_all(fd, buf, (size_t)r)) break;
	}
	close(f);
}

/* the files a trainer may fetch, by exact name: nothing else in the directory leaves it
 * (pending deltas above all), and no path is ever built from what a client sent */
static const char *served(const char *path) {
	static const char *whole[] = { "chain.cfg", "genesis.mlf2", "genesis.opt", "ids.bin", "tip.mlf2", "round" };
	static char blk[32];
	if (path[0] != '/') return NULL;
	for (size_t i = 0; i < sizeof whole / sizeof *whole; i++) if (!strcmp(path + 1, whole[i])) return whole[i];
	if (strncmp(path, "/blocks/", 8) || strlen(path) != 18 || strcmp(path + 14, ".blk")) return NULL;
	for (int i = 8; i < 14; i++) if (!isdigit((unsigned char)path[i])) return NULL;
	snprintf(blk, sizeof blk, "blocks/%.6s.blk", path + 8);
	return blk;
}

/* a whole small file into memory */
static unsigned char *slurp(const char *path, size_t *len) {
	FILE *f = fopen(path, "rb");
	struct stat st;
	unsigned char *b = NULL;
	if (f && fstat(fileno(f), &st) == 0 && st.st_size >= 0 && (b = malloc((size_t)st.st_size + 1)) &&
	    fread(b, 1, (size_t)st.st_size, f) == (size_t)st.st_size) *len = (size_t)st.st_size;
	else { free(b); b = NULL; }
	if (f) fclose(f);
	return b;
}

static int count_pending(const char *dir) {
	char path[PATHN];
	int n = 0;
	DIR *dd = pj(path, dir, "pending") ? NULL : opendir(path);
	struct dirent *e;
	while (dd && (e = readdir(dd))) n += strlen(e->d_name) == 68 && !strcmp(e->d_name + 64, ".dlt");
	if (dd) closedir(dd);
	return n;
}

/* a delta's upload into pending/.up-PID (path), size-checked; 0, or -1 once answered */
static int spool(int fd, const char *dir, const char *body, size_t have, long long len, char *path) {
	char gp[PATHN], name[64];
	struct stat st;
	/* no delta is bigger than every weight as a float, which the genesis file is at least */
	if (pj(gp, dir, "genesis.mlf2") || stat(gp, &st) != 0) { say(fd, 503, "the chain is unreadable here\n"); return -1; }
	if (len <= 0) { say(fd, 400, "a delta needs a Content-Length\n"); return -1; }
	if (len > (long long)st.st_size + 65536) { say(fd, 413, "bigger than any delta of this model\n"); return -1; }
	/* the disk is the box's, shared with whatever else it runs: a delta only while
	 * SPEL_KEEP_FREE megabytes (2048 unless set) stay free after it */
	const char *kf = getenv("SPEL_KEEP_FREE");
	long long mb = kf && *kf ? strtoll(kf, NULL, 10) : 2048;
	if (mb < 0) mb = 0;
	if (mb > (1LL << 40)) mb = 1LL << 40;
	struct statvfs fs;
	if (statvfs(dir, &fs) == 0 && (unsigned long long)fs.f_bavail * fs.f_frsize < (unsigned long long)len + ((unsigned long long)mb << 20)) {
		say(fd, 503, "this coordinator's disk is full for now: next round\n"); return -1;
	}
	snprintf(name, sizeof name, "pending/.up-%ld", (long)getpid());
	if (pj(path, dir, name)) { say(fd, 503, "path too long\n"); return -1; }
	FILE *f = fopen(path, "wb");
	if (!f) { say(fd, 503, "could not keep it\n"); return -1; }
	long long got = 0;
	int ok = 1;
	if (have > (size_t)len) have = (size_t)len;
	if (have) { ok = fwrite(body, 1, have, f) == have; got = (long long)have; }
	char buf[65536];
	while (ok && got < len) {
		size_t want = (size_t)(len - got) < sizeof buf ? (size_t)(len - got) : sizeof buf;
		ssize_t r = read(fd, buf, want);
		if (r < 0 && errno == EINTR) continue;
		if (r <= 0) { ok = 0; break; }
		ok = fwrite(buf, 1, (size_t)r, f) == (size_t)r;
		got += r;
	}
	if (fclose(f) != 0) ok = 0;
	if (!ok) { remove(path); say(fd, 400, "refused: the upload stopped short\n"); return -1; }
	return 0;
}

static void take_delta(int fd, const char *dir, const char *body, size_t have, long long len) {
	char path[PATHN], hex[65], line[160];
	if (count_pending(dir) >= NET_PENDING) { say(fd, 503, "this round is full\n"); return; }
	if (spool(fd, dir, body, have, len, path)) return;
	const char *why = "unreadable";
	if (ch_submit(dir, path, hex, &why) == 0) {
		snprintf(line, sizeof line, "%s\n", hex);
		say(fd, 200, line);
		return;
	}
	remove(path);
	snprintf(line, sizeof line, "refused: %s\n", why);
	say(fd, 400, line);
}

static int run(char *const argv[]);
static void join_url(char *out, size_t cap, const char *url, const char *name);
/* a relay: the delta (or verdict) goes on to its coordinator at the same path, and the coordinator's answer comes back */
static void pass_delta(int fd, const srv *v, const char *where, const char *body, size_t have, long long len) {
	char path[PATHN], ans[PATHN], hdr[PATHN], data[PATHN + 8], u[640], name[64];
	if (spool(fd, v->dir, body, have, len, path)) return;
	snprintf(name, sizeof name, "pending/.ans-%ld", (long)getpid());
	if (pj(ans, v->dir, name)) { remove(path); say(fd, 503, "path too long\n"); return; }
	snprintf(name, sizeof name, "pending/.hdr-%ld", (long)getpid());
	if (pj(hdr, v->dir, name)) { remove(path); say(fd, 503, "path too long\n"); return; }
	snprintf(data, sizeof data, "@%s", path);
	join_url(u, sizeof u, v->upstream, where);
	char *argv[] = { "curl", "-sS", "--connect-timeout", "20", "-H", "Expect:", "-H", "Content-Type: application/octet-stream",
	                 "--data-binary", data, "-D", hdr, "-o", ans, u, NULL };
	int code = 0;
	size_t al = 0;
	unsigned char *a = NULL;
	if (run(argv) == 0) {
		FILE *h = fopen(hdr, "r");
		if (h) { if (fscanf(h, "HTTP/%*s %d", &code) != 1) code = 0; fclose(h); }
		a = slurp(ans, &al);
	}
	remove(path); remove(ans); remove(hdr);
	if (code == 200 || code == 400 || code == 413 || code == 503) answer(fd, code, "text/plain; charset=utf-8", a ? (const char *)a : "", a ? al : 0);
	else say(fd, 503, "the coordinator cannot be reached from this relay\n");
	free(a);
}
/* a relay: what only the coordinator has (the closed round, its deltas, a key's checks), fetched for the asker */
static int fetch(const char *url, const char *remote, const char *dir, const char *name, int quiet);
static void pass_get(int fd, const srv *v, const char *remote) {
	char name[64], path[PATHN];
	snprintf(name, sizeof name, "pending/.get-%ld", (long)getpid());
	if (pj(path, v->dir, name)) { say(fd, 503, "path too long\n"); return; }
	if (fetch(v->upstream, remote, v->dir, name, 1)) { say(fd, 404, "not here\n"); return; }
	send_file(fd, path, "application/octet-stream");
	remove(path);
}

/* ── checks: a closed round's deltas, checked by trainers ──────────────────── */

/* the closed round (DIR/round), when there is one: its parent and deltas' commits and keys */
typedef struct { unsigned char parent[32], seed[32]; int n; unsigned char (*cm)[32], (*ks)[32]; } round_t;
static void round_free(round_t *r) { free(r->cm); free(r->ks); memset(r, 0, sizeof *r); }
static int hex32(const char *s, unsigned char out[32]) {
	if (strlen(s) < 64) return -1;
	for (int i = 0; i < 32; i++) {
		unsigned v;
		if (sscanf(s + 2 * i, "%2x", &v) != 1 || !isxdigit((unsigned char)s[2 * i]) || !isxdigit((unsigned char)s[2 * i + 1])) return -1;
		out[i] = (unsigned char)v;
	}
	return 0;
}
static int round_load(const char *dir, round_t *r) {
	char path[PATHN], line[300], a[80], b[80];
	memset(r, 0, sizeof *r);
	FILE *f = pj(path, dir, "round") ? NULL : fopen(path, "r");
	if (!f) return -1;
	int cap = 0, ok = fgets(line, sizeof line, f) && !strcmp(line, "spelaeum round\n");
	while (ok && fgets(line, sizeof line, f)) {
		if (sscanf(line, "parent %79s", a) == 1) ok = !hex32(a, r->parent);
		else if (sscanf(line, "seed %79s", a) == 1) ok = !hex32(a, r->seed);
		else if (sscanf(line, "delta %79s %79s", a, b) == 2) {
			if (r->n == cap) {
				cap = cap ? cap * 2 : 16;
				void *x = realloc(r->cm, (size_t)cap * 32), *y = x ? realloc(r->ks, (size_t)cap * 32) : NULL;
				if (x) r->cm = x;
				if (y) r->ks = y;
				if (!x || !y) { ok = 0; break; }
			}
			ok = !hex32(a, r->cm[r->n]) && !hex32(b, r->ks[r->n]);
			r->n++;
		}
	}
	fclose(f);
	if (!ok || !r->n) { round_free(r); return -1; }
	return 0;
}
static int round_at(const round_t *r, const unsigned char commit[32]) {
	for (int i = 0; i < r->n; i++) if (!memcmp(r->cm[i], commit, 32)) return i;
	return -1;
}
/* the verdicts kept for delta i, and whether key gave one of them */
static int verdicts(const char *dir, const unsigned char commit[32], const unsigned char key[32], int *from_key) {
	char path[PATHN], name[96], hx[65];
	sha256_hex(commit, hx);   /* hex, nothing hashed */
	snprintf(name, sizeof name, "pending/%s.vrd", hx);
	if (from_key) *from_key = 0;
	FILE *f = pj(path, dir, name) ? NULL : fopen(path, "rb");
	if (!f) return 0;
	ch_verdict v;
	int n = 0;
	while (fread(&v, sizeof v, 1, f) == 1) { n++; if (key && from_key && !memcmp(v.key, key, 32)) *from_key = 1; }
	fclose(f);
	return n;
}
/* every delta of the round has need verdicts, or could not get them from the checkers there are */
static int all_checked(const char *dir, int need) {
	round_t r;
	if (round_load(dir, &r)) return 1;
	int done = 1;
	for (int i = 0; done && i < r.n; i++) {
		if (verdicts(dir, r.cm[i], NULL, NULL) >= need) continue;
		char path[PATHN], line[160];   /* checkers who could still check this one: not its own key, not barred */
		int can = 0;
		FILE *f = pj(path, dir, "checkers") ? NULL : fopen(path, "r");
		while (f && can < need && fgets(line, sizeof line, f)) {
			unsigned char k[32];
			if (!hex32(line, k) && memcmp(k, r.ks[i], 32) && ch_checker(dir, k)) can++;
		}
		if (f) fclose(f);
		done = can < need;
	}
	round_free(&r);
	return done;
}
/* GET /checks/KEY: the deltas this key should check, the least checked first, NET_CHECKS at most */
static void checks_for(int fd, const char *dir, const char *keyhex) {
	unsigned char key[32];
	round_t r;
	if (strlen(keyhex) != 64 || hex32(keyhex, key)) { say(fd, 404, "not here\n"); return; }
	if (round_load(dir, &r)) { say(fd, 404, "no round is being checked\n"); return; }
	char out[NET_CHECKS * 66 + 1], hx[65];
	int k = 0, got = 0;
	out[0] = 0;
	if (ch_checker(dir, key)) {
		int *cnt = malloc((size_t)r.n * sizeof *cnt), *order = malloc((size_t)r.n * sizeof *order), m = 0;
		for (int i = 0; cnt && order && i < r.n; i++) {
			int mine = 0;
			cnt[i] = verdicts(dir, r.cm[i], key, &mine);
			if (!mine && memcmp(r.ks[i], key, 32)) order[m++] = i;
		}
		for (int a = 1; cnt && order && a < m; a++)   /* fewest verdicts first; ties spread by key */
			for (int b = a; b > 0; b--) {
				int x = order[b - 1], y = order[b];
				uint64_t hxk = 0, hyk = 0;
				for (int j = 0; j < 8; j++) { hxk = hxk << 8 | (r.cm[x][j] ^ key[j]); hyk = hyk << 8 | (r.cm[y][j] ^ key[j]); }
				if (cnt[x] < cnt[y] || (cnt[x] == cnt[y] && hxk <= hyk)) break;
				order[b - 1] = y; order[b] = x;
			}
		for (int a = 0; cnt && order && a < m && got < NET_CHECKS; a++, got++) {
			sha256_hex(r.cm[order[a]], hx);
			k += snprintf(out + k, sizeof out - (size_t)k, "%s\n", hx);
		}
		free(cnt); free(order);
	}
	round_free(&r);
	answer(fd, 200, "text/plain; charset=utf-8", out, (size_t)k);
}
/* GET /pending/COMMIT.dlt: a delta of the closed round, never before it closes */
static void round_delta(int fd, const char *dir, const char *path) {
	unsigned char commit[32];
	round_t r;
	char fp[PATHN], name[96];
	if (strlen(path) != 9 + 64 + 4 || strcmp(path + 73, ".dlt") || hex32(path + 9, commit)) { say(fd, 404, "not here\n"); return; }
	if (round_load(dir, &r)) { say(fd, 404, "not here\n"); return; }
	int at = round_at(&r, commit);
	round_free(&r);
	snprintf(name, sizeof name, "pending/%.64s.dlt", path + 9);
	if (at < 0 || pj(fp, dir, name)) { say(fd, 404, "not here\n"); return; }
	send_file(fd, fp, "application/octet-stream");
}
/* POST /verdict: one ch_verdict, from a checker, for a delta of the round not its own */
static void take_verdict(int fd, const char *dir, const char *body, size_t have, long long len) {
	ch_verdict v;
	if (len != (long long)sizeof v) { say(fd, 400, "a verdict is 136 bytes\n"); return; }
	size_t got = have > sizeof v ? sizeof v : have;
	memcpy(&v, body, got);
	while (got < sizeof v) {
		ssize_t r = read(fd, (char *)&v + got, sizeof v - got);
		if (r < 0 && errno == EINTR) continue;
		if (r <= 0) { say(fd, 400, "the verdict stopped short\n"); return; }
		got += (size_t)r;
	}
	round_t r;
	if (round_load(dir, &r)) { say(fd, 400, "refused: no round is being checked\n"); return; }
	int at = round_at(&r, v.commit), mine = 0;
	const char *why = at < 0 ? "that delta is not in the round" : !memcmp(r.ks[at], v.key, 32) ? "a key does not check its own delta" :
	                  !ch_checker(dir, v.key) ? "not a checker here: a key checks once it has a passed job" :
	                  !ch_verdict_ok(r.parent, &v) ? "its signature is not its key's" : NULL;
	round_free(&r);
	if (!why && (verdicts(dir, v.commit, v.key, &mine), mine)) why = "this key already checked that delta";
	char path[PATHN], name[96], hx[65], line[160];
	if (!why) {
		sha256_hex(v.commit, hx);   /* hex, nothing hashed */
		snprintf(name, sizeof name, "pending/%s.vrd", hx);
		int f = pj(path, dir, name) ? -1 : open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
		int ok = f >= 0 && write(f, &v, sizeof v) == (ssize_t)sizeof v;
		if (f >= 0) close(f);
		if (ok) { say(fd, 200, "taken\n"); return; }
		why = "could not keep it";
	}
	snprintf(line, sizeof line, "refused: %s\n", why);
	say(fd, 400, line);
}

static void handle(int fd, const srv *v, long long seal) {
	const char *dir = v->dir;
	struct timeval tv = { 20, 0 };   /* a client that stops talking is dropped */
	setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
	setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
	char h[HEADN + 1];
	size_t n = 0;
	char *end = NULL;
	while (n < HEADN) {
		ssize_t r = read(fd, h + n, HEADN - n);
		if (r < 0 && errno == EINTR) continue;
		if (r <= 0) return;
		n += (size_t)r; h[n] = 0;
		if ((end = strstr(h, "\r\n\r\n"))) break;
	}
	if (!end) { say(fd, 400, "headers too long\n"); return; }
	size_t body_at = (size_t)(end - h) + 4;
	*end = 0;
	char method[8], path[160];
	if (sscanf(h, "%7s %159s", method, path) != 2) { say(fd, 400, "not a request\n"); return; }
	char *q = strchr(path, '?');
	if (q) *q = 0;
	long long len = -1;
	for (char *l = strstr(h, "\r\n"); l; l = strstr(l + 2, "\r\n"))
		if (!strncasecmp(l + 2, "content-length:", 15)) len = atoll(l + 17);

	if (!strcmp(method, "OPTIONS")) { answer(fd, 204, "text/plain", "", 0); return; }
	if (!strcmp(method, "POST")) {
		if (strcmp(path, "/submit") && strcmp(path, "/verdict")) { say(fd, 404, "not here\n"); return; }
		char rp[PATHN];
		int closed = pj(rp, dir, "round") == 0 && access(rp, F_OK) == 0;
		if (!strcmp(path, "/verdict") && !v->upstream) { take_verdict(fd, dir, h + body_at, n - body_at, len); return; }
		if (v->upstream) pass_delta(fd, v, path + 1, h + body_at, n - body_at, len);
		else if (closed) say(fd, 503, "the round is closed while its deltas are checked: the next one opens with the block\n");
		else take_delta(fd, dir, h + body_at, n - body_at, len);
		return;
	}
	if (strcmp(method, "GET")) { say(fd, 405, "GET and POST only\n"); return; }
	if (!strcmp(path, "/tip") && v->upstream) {   /* the coordinator's tip as this copy last reached it, on its clock */
		char tp[PATHN], out[512], line[160];
		long long off = 0;
		FILE *f = pj(tp, dir, "upstream.tip") ? NULL : fopen(tp, "r");
		if (!f || !fgets(line, sizeof line, f) || sscanf(line, "offset %lld", &off) != 1) { if (f) fclose(f); say(fd, 503, "this relay has not reached its coordinator yet\n"); return; }
		int k = 0;
		while (k < (int)sizeof out - 200 && fgets(line, sizeof line, f)) k += snprintf(out + k, sizeof out - (size_t)k, "%s", line);
		fclose(f);
		snprintf(out + k, sizeof out - (size_t)k, "now %lld\n", (long long)time(NULL) + off);
		say(fd, 200, out);
		return;
	}
	if (!strcmp(path, "/tip")) {
		uint64_t height; unsigned char hash[32];
		if (ch_tip(dir, &height, hash)) { say(fd, 503, "the chain is unreadable here\n"); return; }
		char hx[65], out[1024], mp[PATHN];
		static const char hexd[] = "0123456789abcdef";
		for (int i = 0; i < 32; i++) { hx[2 * i] = hexd[hash[i] >> 4]; hx[2 * i + 1] = hexd[hash[i] & 15]; }
		hx[64] = 0;
		int k = snprintf(out, sizeof out, "height %llu\nhash %s\nseal %lld\nnow %lld\nround %d\n", (unsigned long long)height, hx, seal, (long long)time(NULL), v->round_s);
		if (v->files) k += snprintf(out + k, sizeof out - (size_t)k, "files %s\n", v->files);   /* url_ok: 512 bytes at most */
		if (pj(mp, dir, "round") == 0 && access(mp, F_OK) == 0) k += snprintf(out + k, sizeof out - (size_t)k, "phase checking\n");
		/* how far the mirror has got: DIR/mirrored, written by whatever uploads it */
		FILE *mf = pj(mp, dir, "mirrored") ? NULL : fopen(mp, "r");
		unsigned long long mh;
		if (mf) { if (fscanf(mf, "%llu", &mh) == 1) snprintf(out + k, sizeof out - (size_t)k, "mirrored %llu\n", mh); fclose(mf); }
		say(fd, 200, out);
		return;
	}
	if (!strcmp(path, "/status.json")) {
		char *buf = NULL; size_t bl = 0;
		FILE *m = open_memstream(&buf, &bl);
		int bad = !m || ch_status_json(dir, m, 64);
		if (m) fclose(m);
		if (bad) say(fd, 503, "the chain is unreadable here\n");
		else answer(fd, 200, "application/json", buf, bl);
		free(buf);
		return;
	}
	if (!strncmp(path, "/heads/", 7)) {   /* head parts from a height up, NET_HEADS at most: how a copy starts late */
		char *e = NULL;
		unsigned long long from = strtoull(path + 7, &e, 10);
		uint64_t height; unsigned char hash[32];
		if (!e || *e || e == path + 7 || from < 1) { say(fd, 404, "not here\n"); return; }
		if (ch_tip(dir, &height, hash)) { say(fd, 503, "the chain is unreadable here\n"); return; }
		if (from > height) { say(fd, 404, "no such block yet\n"); return; }
		char *buf = NULL; size_t bl = 0;
		FILE *m = open_memstream(&buf, &bl);
		int bad = !m;
		for (uint64_t b = from; !bad && b <= height && b < from + NET_HEADS; b++) {
			unsigned char *hp; size_t hl;
			if (ch_head_part(dir, b, &hp, &hl)) { bad = 1; break; }
			bad = fwrite(hp, 1, hl, m) != hl;
			free(hp);
		}
		if (m) fclose(m);
		if (bad) say(fd, 503, "the chain is unreadable here\n");
		else answer(fd, 200, "application/octet-stream", buf, bl);
		free(buf);
		return;
	}
	int roundish = !strncmp(path, "/checks/", 8) || !strncmp(path, "/pending/", 9) || !strcmp(path, "/round");
	if (roundish && v->upstream) {   /* names checked here, so only these paths ever reach the coordinator */
		const char *x = path + 1;
		int ok = strlen(x) < 96;
		for (const char *c = x; ok && *c; c++) ok = isalnum((unsigned char)*c) || *c == '/' || *c == '.';
		if (ok) pass_get(fd, v, x); else say(fd, 404, "not here\n");
		return;
	}
	if (!strncmp(path, "/checks/", 8)) { checks_for(fd, dir, path + 8); return; }
	if (!strncmp(path, "/pending/", 9)) { round_delta(fd, dir, path); return; }
	const char *name = served(path);
	char fp[PATHN];
	if (!name || pj(fp, dir, name)) { say(fd, 404, "not here\n"); return; }
	send_file(fd, fp, "application/octet-stream");
}

/* the --after program for the block just sealed, in the background: its pid, or 0 */
static pid_t run_after(const char *after, const char *dir, int ls) {
	uint64_t h; unsigned char hash[32];
	char hs[24];
	if (ch_tip(dir, &h, hash)) return 0;
	snprintf(hs, sizeof hs, "%llu", (unsigned long long)h);
	fflush(stdout); fflush(stderr);
	pid_t pid = fork();
	if (pid == 0) { close(ls); execl(after, after, dir, hs, (char *)NULL); perror(after); _exit(127); }
	return pid > 0 ? pid : 0;
}

static int listen_on(const char *bind_addr, int port) {
	signal(SIGPIPE, SIG_IGN);
	int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
	struct sockaddr_in a;
	memset(&a, 0, sizeof a);
	a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
	if (ls < 0 || inet_pton(AF_INET, bind_addr ? bind_addr : "127.0.0.1", &a.sin_addr) != 1) { fprintf(stderr, "serve: bad address\n"); if (ls >= 0) close(ls); return -1; }
	setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
	if (bind(ls, (struct sockaddr *)&a, sizeof a) != 0 || listen(ls, 64) != 0) { perror("serve: bind"); close(ls); return -1; }
	return ls;
}
/* one connection, in its own process */
static void serve_one(int ls, int *live, const srv *v, long long seal) {
	int fd = accept(ls, NULL, NULL);
	if (fd < 0) return;
	if (*live >= NET_CONNS) { say(fd, 503, "busy\n"); close(fd); return; }
	pid_t pid = fork();
	if (pid == 0) { close(ls); handle(fd, v, seal); shutdown(fd, SHUT_WR); close(fd); _exit(0); }
	if (pid > 0) (*live)++;
	close(fd);
}

int net_serve(const char *dir, const char *bind_addr, int port, int round_s, const char *files, const char *after, int ready_fd) {
	net_checks none = { 0, 0, 0 };
	return net_serve_checked(dir, bind_addr, port, round_s, files, after, &none, ready_fd);
}

int net_serve_checked(const char *dir, const char *bind_addr, int port, int round_s, const char *files, const char *after,
                      const net_checks *cx, int ready_fd) {
	uint64_t height; unsigned char hash[32];
	if (ch_tip(dir, &height, hash)) return -1;
	if (round_s < 1 || port < 1 || port > 65535) { fprintf(stderr, "serve: a round of at least 1 second, a port 1-65535\n"); return -1; }
	if (files && !url_ok(files)) { fprintf(stderr, "serve: --files names the mirror trainers fetch from: http:// or https://, without spaces\n"); return -1; }
	if (after && access(after, X_OK) != 0) { fprintf(stderr, "serve: --after %s is not a program this can run\n", after); return -1; }
	int ls = listen_on(bind_addr, port);
	if (ls < 0) return -1;
	srv v = { dir, files, after, NULL, round_s, cx->need, cx->spot, cx->window > 0 ? cx->window : NET_CHECK_WINDOW };
	char rp[PATHN];
	if (pj(rp, dir, "round")) { close(ls); return -1; }
	if (access(rp, F_OK) == 0) remove(rp);   /* a round left closed by a coordinator that stopped: reopen it */
	fprintf(stderr, "serve: %s on %s:%d, a round every %d s, height %llu%s%s\n", dir, bind_addr ? bind_addr : "127.0.0.1", port, round_s,
	        (unsigned long long)height, files ? ", mirror " : "", files ? files : "");
	if (ready_fd >= 0) { if (write(ready_fd, "r", 1) != 1) perror("serve: ready"); }
	long long seal = (long long)time(NULL) + round_s;
	int live = 0, owed = 0;   /* owed: a block sealed while --after still ran */
	pid_t hook = 0;
	for (;;) {
		pid_t w;
		while ((w = waitpid(-1, NULL, WNOHANG)) > 0) {
			if (w != hook) { if (live > 0) live--; continue; }
			hook = owed ? run_after(after, dir, ls) : 0;
			owed = 0;
		}
		long long now = (long long)time(NULL);
		int closed = access(rp, F_OK) == 0;
		if (v.need > 0 && !closed && now >= seal) {   /* close the round, and let trainers check it */
			int cr = ch_close(dir);
			if (cr == 0) { seal = now + v.check_s; fprintf(stderr, "serve: the round is closed, checked until the block or %d s\n", v.check_s); continue; }
			if (cr < 0) fprintf(stderr, "serve: could not close the round\n");
			seal = now + round_s;
		} else if (now >= seal || (v.need > 0 && closed && all_checked(dir, v.need))) {
			/* uploads still arriving land in the next round, or are stale there and set aside */
			int taken = 0, sr = v.need > 0 ? ch_seal_checked(dir, v.need, v.spot, &taken) : ch_seal(dir);
			if (sr < 0) fprintf(stderr, "serve: the seal failed, the round's deltas stay pending\n");
			else if (sr == 0 && after) { if (hook) owed = 1; else hook = run_after(after, dir, ls); }
			if (sr == 0 && v.need > 0) fprintf(stderr, "serve: %d deltas taken from trainers' checks\n", taken);
			if (access(rp, F_OK) == 0) remove(rp);
			seal = (long long)time(NULL) + round_s;
		}
		struct pollfd pf = { ls, POLLIN, 0 };
		if (poll(&pf, 1, 500) > 0) serve_one(ls, &live, &v, seal);
	}
}

/* a relay's copy brought to its coordinator's tip, and that tip kept for /tip with
 * the coordinator's clock as an offset from this one's */
static int relay_sync(const char *url, const char *dir) {
	net_tip t;
	uint64_t h; unsigned char hash[32];
	char hx[65], path[PATHN], tmp[PATHN + 8];
	if (net_tip_get(url, dir, &t)) return -1;
	int same = ch_tip(dir, &h, hash) == 0 && h == t.height && (sha256_hex(hash, hx), !strcmp(hx, t.hash));   /* hex, nothing hashed */
	long long off = t.now - (long long)time(NULL);
	if (!same && net_sync(url, NULL, dir, &t, 0)) return -1;
	if (pj(path, dir, "upstream.tip")) return -1;
	snprintf(tmp, sizeof tmp, "%s.part", path);
	FILE *f = fopen(tmp, "w");
	if (!f) return -1;
	fprintf(f, "offset %lld\nheight %llu\nhash %s\nseal %lld\nround %d\n%s", off, (unsigned long long)t.height, t.hash, t.seal, t.round,
	        t.checking ? "phase checking\n" : "");
	if (fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); return -1; }
	return 0;
}

int net_relay(const char *url, const char *dir, const char *bind_addr, int port, int ready_fd) {
	if (!url_ok(url)) { fprintf(stderr, "relay: a url is http:// or https://, without spaces\n"); return -1; }
	if (port < 1 || port > 65535) { fprintf(stderr, "relay: a port 1-65535\n"); return -1; }
	if (mkdir(dir, 0755) != 0 && errno != EEXIST) { perror(dir); return -1; }
	if (relay_sync(url, dir)) { fprintf(stderr, "relay: could not bring %s up to %s\n", dir, url); return -1; }
	int ls = listen_on(bind_addr, port);
	if (ls < 0) return -1;
	fprintf(stderr, "relay: %s on %s:%d for %s, checked every %d s\n", dir, bind_addr ? bind_addr : "127.0.0.1", port, url, NET_RELAY_POLL);
	if (ready_fd >= 0) { if (write(ready_fd, "r", 1) != 1) perror("relay: ready"); }
	srv v = { dir, NULL, NULL, url, 0, 0, 0, 0 };
	int live = 0;
	pid_t sync = 0;
	long long next = (long long)time(NULL) + NET_RELAY_POLL;
	for (;;) {
		pid_t w;
		while ((w = waitpid(-1, NULL, WNOHANG)) > 0) { if (w == sync) sync = 0; else if (live > 0) live--; }
		if (!sync && (long long)time(NULL) >= next) {   /* the copy kept up in a child, never two at once */
			fflush(stdout); fflush(stderr);
			pid_t pid = fork();
			if (pid == 0) { close(ls); _exit(relay_sync(url, dir) ? 1 : 0); }
			if (pid > 0) sync = pid;
			next = (long long)time(NULL) + NET_RELAY_POLL;
		}
		struct pollfd pf = { ls, POLLIN, 0 };
		if (poll(&pf, 1, 500) > 0) serve_one(ls, &live, &v, 0);
	}
}

/* ── a trainer's side ─────────────────────────────────────────────────────── */

static void join_url(char *out, size_t cap, const char *url, const char *name) {
	size_t n = strlen(url);
	while (n && url[n - 1] == '/') n--;
	snprintf(out, cap, "%.*s/%s", (int)n, url, name);
}
static int run(char *const argv[]) {
	pid_t pid = fork();
	if (pid < 0) return -1;
	if (pid == 0) { execvp(argv[0], argv); perror("curl"); _exit(127); }
	int st = 0;
	while (waitpid(pid, &st, 0) < 0) if (errno != EINTR) return -1;
	return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? 0 : -1;
}
/* url/name into dir/name, through a .part and a rename; quiet = a 404 is not worth a line */
static int fetch(const char *url, const char *remote, const char *dir, const char *name, int quiet) {
	char u[640], dest[PATHN], part[PATHN + 8];
	if (pj(dest, dir, name)) return -1;
	snprintf(part, sizeof part, "%s.part", dest);
	join_url(u, sizeof u, url, remote);
	/* -L: a mirror on Hugging Face answers a file with a redirect to its CDN */
	char *argv[] = { "curl", "-f", "-L", "-sS", "--retry", "2", "--connect-timeout", "20", "-o", part, u, NULL };
	char *argq[] = { "curl", "-f", "-L", "-s", "--connect-timeout", "20", "-o", part, u, NULL };
	if (run(quiet ? argq : argv) != 0) { remove(part); return -1; }
	return rename(part, dest);
}

int net_tip_get(const char *url, const char *dir, net_tip *t) {
	char path[PATHN], key[16], val[520];
	if (!url_ok(url)) { fprintf(stderr, "net: a url is http:// or https://, without spaces\n"); return -1; }
	if (fetch(url, "tip", dir, "tip.remote", 0) || pj(path, dir, "tip.remote")) return -1;
	FILE *f = fopen(path, "r");
	if (!f) return -1;
	memset(t, 0, sizeof *t);
	int got = 0;
	while (fscanf(f, "%15s %519s", key, val) == 2) {
		if (!strcmp(key, "height")) { t->height = strtoull(val, NULL, 10); got |= 1; }
		else if (!strcmp(key, "hash") && strlen(val) == 64) { memcpy(t->hash, val, 65); got |= 2; }
		else if (!strcmp(key, "seal")) t->seal = atoll(val);
		else if (!strcmp(key, "now")) t->now = atoll(val);
		else if (!strcmp(key, "round")) t->round = atoi(val);
		else if (!strcmp(key, "files") && files_ok(val, 0)) snprintf(t->files, sizeof t->files, "%s", val);
		else if (!strcmp(key, "mirrored")) { t->mirrored = strtoull(val, NULL, 10); t->has_mirrored = 1; }
		else if (!strcmp(key, "phase")) t->checking = !strcmp(val, "checking");
	}
	fclose(f);
	remove(path);
	if (got != 3) { fprintf(stderr, "net: %s did not answer like a coordinator\n", url); return -1; }
	return 0;
}

/* a big file: from the mirror (as mname) when there is one and it has it, else the coordinator's (as rname) */
static int fetch_big(const char *url, const char *files, const char *mname, const char *rname, const char *dir, const char *name,
                     int quiet, int *nf, int *nu) {
	if (*files && fetch(files, mname, dir, name, 1) == 0) { (*nf)++; return 0; }
	if (fetch(url, rname, dir, name, quiet)) return -1;
	(*nu)++;
	return 0;
}
/* the height of the snapshot a mirror holds: its tip file, "HEIGHT HASH" as the chain writes it */
static int mirror_tip(const char *files, const char *dir, uint64_t *h) {
	char path[PATHN], hx[80];
	unsigned long long v;
	if (fetch(files, "tip", dir, "tip.mirror", 1) || pj(path, dir, "tip.mirror")) return -1;
	FILE *f = fopen(path, "r");
	int ok = f && fscanf(f, "%llu %79s", &v, hx) == 2 && strlen(hx) == 64;
	if (f) fclose(f);
	remove(path);
	if (ok) *h = v;
	return ok ? 0 : -1;
}
/* the coordinator's heads from block `from` through upto into dir, as pruned blocks */
static int put_heads(const char *url, const char *dir, uint64_t from, uint64_t upto) {
	char path[PATHN], rem[48];
	while (from <= upto) {
		unsigned char *b = NULL;
		size_t bl = 0;
		uint64_t got = 0;
		snprintf(rem, sizeof rem, "heads/%llu", (unsigned long long)from);
		int bad = fetch(url, rem, dir, "heads.part", 0) || pj(path, dir, "heads.part") || !(b = slurp(path, &bl)) ||
		          ch_put_heads(dir, b, bl, from, upto, &got) || got == 0;
		free(b);
		if (pj(path, dir, "heads.part") == 0) remove(path);
		if (bad) { fprintf(stderr, "net: could not fetch the heads from block %llu\n", (unsigned long long)from); return -1; }
		from += got;
	}
	return 0;
}
/* a coordinator that says how far its mirror has got: wait a while for it to reach the tip,
 * rather than have every trainer take the newest block from the coordinator */
static void mirror_wait(const char *url, const char *dir, net_tip *tip) {
	const char *e = getenv("SPEL_MIRROR_WAIT");
	long long until = (long long)time(NULL) + (e ? atol(e) : NET_MIRROR_WAIT);
	int said = 0;
	while (tip->has_mirrored && tip->mirrored < tip->height && (long long)time(NULL) < until) {
		if (!said++) fprintf(stderr, "net: waiting for the mirror to reach block %llu\n", (unsigned long long)tip->height);
		long long left = until - (long long)time(NULL);
		sleep(left > 5 ? 5 : left > 0 ? (unsigned)left : 1);
		net_tip n;
		if (net_tip_get(url, dir, &n) == 0) *tip = n;
	}
}

int net_sync(const char *url, const char *files, const char *dir, net_tip *t, int whole) {
	char path[PATHN], name[32], mname[48], mir[520];
	net_tip tip;
	int nf = 0, nu = 0;
	if (!url_ok(url)) { fprintf(stderr, "net: a url is http:// or https://, without spaces\n"); return -1; }
	if (files && *files && !files_ok(files, 1)) { fprintf(stderr, "net: a mirror is http://, https:// or file://, without spaces\n"); return -1; }
	if (mkdir(dir, 0755) != 0 && errno != EEXIST) { perror(dir); return -1; }
	if (pj(path, dir, "blocks") || (mkdir(path, 0755) != 0 && errno != EEXIST)) return -1;
	if (pj(path, dir, "pending") || (mkdir(path, 0755) != 0 && errno != EEXIST)) return -1;
	if (net_tip_get(url, dir, &tip)) return -1;
	snprintf(mir, sizeof mir, "%s", files ? files : tip.files);   /* this machine's mirror, else the coordinator's */
	if (pj(path, dir, "chain.cfg")) return -1;   /* the chain's name: from the coordinator, never a mirror */
	if (access(path, F_OK) != 0 && fetch(url, "chain.cfg", dir, "chain.cfg", 0)) { fprintf(stderr, "net: could not fetch chain.cfg\n"); return -1; }
	static const char *need[] = { "genesis.mlf2", "ids.bin" };
	for (int i = 0; i < 2; i++) {
		if (pj(path, dir, need[i])) return -1;
		if (access(path, F_OK) != 0 && fetch_big(url, mir, need[i], need[i], dir, need[i], 0, &nf, &nu)) { fprintf(stderr, "net: could not fetch %s\n", need[i]); return -1; }
	}
	if (pj(path, dir, "genesis.opt")) return -1;
	if (access(path, F_OK) != 0) fetch_big(url, mir, "genesis.opt", "genesis.opt", dir, "genesis.opt", 1, &nf, &nu);   /* a chain may have none */
	if (ch_start(dir)) { fprintf(stderr, "net: the coordinator's data or genesis does not check out\n"); return -1; }
	/* a copy a few blocks behind fetches them whole and applies them; one further
	 * behind, or new, fetches heads and a model whole: old deltas never travel */
	for (int tries = 0;; tries++) {
		if (tries && net_tip_get(url, dir, &tip)) return -1;
		snprintf(name, sizeof name, "blocks/%06llu.blk", (unsigned long long)tip.height + 1);
		if (pj(path, dir, name)) return -1;
		if (access(path, F_OK) == 0) { fprintf(stderr, "net: %s holds more blocks than the coordinator has: not the same chain any more\n", dir); return -1; }
		uint64_t h0 = 0; unsigned char hh[32];
		if (ch_tip(dir, &h0, hh)) return -1;
		if (h0 > tip.height) { fprintf(stderr, "net: %s is past the coordinator's tip: not the same chain any more\n", dir); return -1; }
		if (*mir && h0 < tip.height) mirror_wait(url, dir, &tip);
		if (whole || tip.height - h0 <= NET_WHOLE) {
			for (uint64_t h = h0 + 1; h <= tip.height; h++) {
				snprintf(name, sizeof name, "blocks/%06llu.blk", (unsigned long long)h);
				mirror_block(mname, sizeof mname, h);
				if (fetch_big(url, mir, mname, name, dir, name, 0, &nf, &nu)) { fprintf(stderr, "net: could not fetch block %llu\n", (unsigned long long)h); return -1; }
			}
			if (ch_advance(dir) == 0) break;
			if (whole) { fprintf(stderr, "net: the coordinator's blocks do not replay\n"); return -1; }
			/* a coordinator that pruned them: heads and the model will do */
			fprintf(stderr, "net: taking the heads and the model instead\n");
		}
		/* a mirror's snapshot between here and the tip: the heads up to it, its model,
		 * then the blocks after it whole. Anything amiss, and the coordinator's model */
		uint64_t s = 0;
		if (*mir && mirror_tip(mir, dir, &s) == 0 && s > h0 && s <= tip.height && put_heads(url, dir, h0 + 1, s) == 0 &&
		    fetch(mir, "tip.mlf2", dir, "tip.new", 1) == 0) {
			nf++;
			if (ch_join(dir, "tip.new") == 0) {
				int ok = 1;
				for (uint64_t h = s + 1; ok && h <= tip.height; h++) {
					snprintf(name, sizeof name, "blocks/%06llu.blk", (unsigned long long)h);
					mirror_block(mname, sizeof mname, h);
					ok = fetch_big(url, mir, mname, name, dir, name, 0, &nf, &nu) == 0;
				}
				if (ok && (s == tip.height || ch_advance(dir) == 0)) break;
				if (ch_tip(dir, &h0, hh)) return -1;   /* joined at s: the rest from there */
			} else fprintf(stderr, "net: the mirror's model is not the one block %llu names: taking the coordinator's\n", (unsigned long long)s);
		}
		if (pj(path, dir, "tip.new") == 0) remove(path);
		if (put_heads(url, dir, h0 + 1, tip.height)) return -1;
		if (fetch(url, "tip.mlf2", dir, "tip.new", 0)) { fprintf(stderr, "net: could not fetch tip.mlf2\n"); return -1; }
		nu++;
		if (ch_join(dir, "tip.new") == 0) break;
		if (pj(path, dir, "tip.new") == 0) remove(path);
		/* sealed between the heads and the model: the next try has one head more */
		if (tries == 2) { fprintf(stderr, "net: the coordinator's heads and model do not agree\n"); return -1; }
	}
	uint64_t h; unsigned char hash[32];
	char hx[65];
	static const char hexd[] = "0123456789abcdef";
	if (ch_tip(dir, &h, hash)) return -1;
	for (int i = 0; i < 32; i++) { hx[2 * i] = hexd[hash[i] >> 4]; hx[2 * i + 1] = hexd[hash[i] & 15]; }
	hx[64] = 0;
	if (h != tip.height || strcmp(hx, tip.hash)) { fprintf(stderr, "net: the blocks replay to another tip than the coordinator names\n"); return -1; }
	if (!whole && ch_prune(dir, NET_WHOLE, 1)) return -1;   /* a trainer's copy never needs old deltas */
	tip.from_files = nf; tip.from_url = nu;
	if (t) *t = tip;
	return 0;
}

int net_check(const char *url, const char *dir, const ch_key *key) {
	char path[PATHN], rpath[PATHN], dpath[PATHN], vpath[PATHN], data[PATHN + 8], u[640], kx[65], rem[96], line[160];
	if (!url_ok(url)) return -1;
	sha256_hex(key->pk, kx);   /* hex, nothing hashed */
	snprintf(rem, sizeof rem, "checks/%s", kx);
	if (fetch(url, "round", dir, "round.remote", 1) || fetch(url, rem, dir, "checks.remote", 1)) return -1;
	if (pj(rpath, dir, "round.remote") || pj(path, dir, "checks.remote") || pj(dpath, dir, "check.dlt") || pj(vpath, dir, "verdict.bin")) return -1;
	FILE *f = fopen(path, "r");
	int sent = 0;
	while (f && fgets(line, sizeof line, f)) {
		line[strcspn(line, "\n")] = 0;
		unsigned char c[32], parent[32];
		ch_verdict v;
		if (strlen(line) != 64 || hex32(line, c)) continue;
		snprintf(rem, sizeof rem, "pending/%s.dlt", line);
		if (fetch(url, rem, dir, "check.dlt", 1) || ch_check(dir, rpath, dpath, parent, &v) || memcmp(v.commit, c, 32)) { remove(dpath); continue; }
		remove(dpath);
		ch_verdict_sign(key, parent, &v);
		FILE *o = fopen(vpath, "wb");
		int ok = o && fwrite(&v, sizeof v, 1, o) == 1;
		if (o && fclose(o) != 0) ok = 0;
		if (!ok) continue;
		snprintf(data, sizeof data, "@%s", vpath);
		join_url(u, sizeof u, url, "verdict");
		char *argv[] = { "curl", "--fail-with-body", "-sS", "--connect-timeout", "20", "-H", "Expect:", "-H", "Content-Type: application/octet-stream",
		                 "--data-binary", data, "-o", "/dev/null", u, NULL };
		if (run(argv) == 0) sent++;
		remove(vpath);
	}
	if (f) fclose(f);
	remove(path); remove(rpath);
	return sent;
}

int net_submit(const char *url, const char *dir, const char *hex) {
	char u[640], path[PATHN], name[96], data[PATHN + 8];
	if (!url_ok(url) || strlen(hex) != 64) return -1;
	snprintf(name, sizeof name, "pending/%s.dlt", hex);
	if (pj(path, dir, name)) return -1;
	snprintf(data, sizeof data, "@%s", path);
	join_url(u, sizeof u, url, "submit");
	/* no Expect: curl would wait a second for a 100 this coordinator never sends */
	char *argv[] = { "curl", "--fail-with-body", "-sS", "--connect-timeout", "20", "-H", "Expect:", "-H", "Content-Type: application/octet-stream",
	                 "--data-binary", data, "-o", "/dev/stderr", u, NULL };
	if (run(argv) != 0) return -1;
	remove(path);
	return 0;
}

/* ── a mirror ───────────────────────────────────────────────────────────── */

/* from into to, through a .part and a rename */
static int copy_to(const char *from, const char *to) {
	char part[PATHN + 8], buf[65536];
	snprintf(part, sizeof part, "%s.part", to);
	FILE *in = fopen(from, "rb"), *out = in ? fopen(part, "wb") : NULL;
	int ok = in && out;
	size_t r;
	while (ok && (r = fread(buf, 1, sizeof buf, in)) > 0) ok = fwrite(buf, 1, r, out) == r;
	if (in && ferror(in)) ok = 0;
	if (in) fclose(in);
	if (out && fclose(out) != 0) ok = 0;
	if (!ok || rename(part, to) != 0) { remove(part); return -1; }
	return 0;
}

int net_mirror(const char *dir, const char *out, uint64_t every, FILE *added) {
	char a[PATHN], b[PATHN], name[48], mname[48];
	uint64_t height, now; unsigned char hash[32], again[32];
	if (ch_tip(dir, &height, hash)) return -1;
	if (every < 1) every = 1;
	if ((mkdir(out, 0755) != 0 && errno != EEXIST) || pj(b, out, "blocks") || (mkdir(b, 0755) != 0 && errno != EEXIST)) { perror(out); return -1; }
	static const char *fixed[] = { "chain.cfg", "genesis.mlf2", "genesis.opt", "ids.bin" };
	for (int i = 0; i < 4; i++) {
		if (pj(a, dir, fixed[i]) || pj(b, out, fixed[i])) return -1;
		if (access(a, F_OK) != 0 || access(b, F_OK) == 0) continue;   /* a chain may have no genesis.opt */
		if (copy_to(a, b)) { fprintf(stderr, "mirror: could not copy %s\n", fixed[i]); return -1; }
		if (added) fprintf(added, "%s\n", fixed[i]);
	}
	for (uint64_t h = 1; h <= height; h++) {
		if (h == 1 || h % 1000 == 0) {
			snprintf(name, sizeof name, "blocks/%03llu", (unsigned long long)(h / 1000));
			if (pj(b, out, name) || (mkdir(b, 0755) != 0 && errno != EEXIST)) { perror(b); return -1; }
		}
		snprintf(name, sizeof name, "blocks/%06llu.blk", (unsigned long long)h);
		mirror_block(mname, sizeof mname, h);
		if (pj(a, dir, name) || pj(b, out, mname)) return -1;
		if (access(b, F_OK) == 0) continue;
		if (copy_to(a, b)) { fprintf(stderr, "mirror: could not copy block %llu\n", (unsigned long long)h); return -1; }
		if (added) fprintf(added, "%s\n", mname);
	}
	/* the snapshot, once the tip is `every` past the mirror's: the model first, then
	 * the tip that names it, and none at all when a seal moved the tip meanwhile */
	unsigned long long v = 0;
	FILE *f = pj(b, out, "tip") ? NULL : fopen(b, "r");
	int have = f && fscanf(f, "%llu", &v) == 1;
	if (f) fclose(f);
	if (have && v > height) { fprintf(stderr, "mirror: %s is at block %llu, past this chain's %llu: not its mirror\n", out, v, (unsigned long long)height); return -1; }
	if (have && height < v + every) return 0;
	if (pj(a, dir, "tip.mlf2") || pj(b, out, "tip.mlf2.new") || copy_to(a, b)) { fprintf(stderr, "mirror: could not copy tip.mlf2\n"); return -1; }
	if (ch_tip(dir, &now, again) || now != height || memcmp(again, hash, 32)) {
		remove(b);
		fprintf(stderr, "mirror: a block was sealed while the model was copied: the next run takes the snapshot\n");
		return 0;
	}
	char c[PATHN];
	if (pj(c, out, "tip.mlf2") || rename(b, c) != 0 || pj(a, dir, "tip") || pj(b, out, "tip") || copy_to(a, b)) { fprintf(stderr, "mirror: could not write the snapshot\n"); return -1; }
	if (added) fprintf(added, "tip.mlf2\ntip\n");
	return 0;
}
