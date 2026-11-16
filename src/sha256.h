/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* sha256.h — FIPS 180-4 SHA-256, for the chain's block and commit hashes. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { uint32_t s[8]; uint64_t len; unsigned char buf[64]; size_t nbuf; } sha256_ctx;

void sha256_init(sha256_ctx *c);
void sha256_add(sha256_ctx *c, const void *data, size_t n);
void sha256_done(sha256_ctx *c, unsigned char out[32]);
void sha256(const void *data, size_t n, unsigned char out[32]);
/* 64 lowercase hex chars and a NUL */
void sha256_hex(const unsigned char h[32], char out[65]);
