/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* ed25519.h — RFC 8032 Ed25519 signatures and the FIPS 180-4 SHA-512 they are made
 * with: a trainer's key, so a delta in someone's name was sent by them. Field and
 * point code after TweetNaCl (public domain): small, constant time where the
 * secret is, slow next to a tuned library and nowhere near the cost of a round. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { uint64_t s[8]; uint64_t len; unsigned char buf[128]; size_t nbuf; } sha512_ctx;
void sha512_init(sha512_ctx *c);
void sha512_add(sha512_ctx *c, const void *data, size_t n);
void sha512_done(sha512_ctx *c, unsigned char out[64]);
void sha512(const void *data, size_t n, unsigned char out[64]);

/* the public key of a 32-byte secret seed */
void ed25519_public(const unsigned char seed[32], unsigned char pk[32]);
void ed25519_sign(unsigned char sig[64], const void *msg, size_t n, const unsigned char seed[32], const unsigned char pk[32]);
/* 0 when sig is pk's signature of msg; -1 otherwise, a key that is not a point included */
int ed25519_verify(const unsigned char sig[64], const void *msg, size_t n, const unsigned char pk[32]);
