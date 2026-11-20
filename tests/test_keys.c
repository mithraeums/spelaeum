/* SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser */
/* test_keys.c — SHA-512 against the FIPS vectors, Ed25519 against RFC 8032's, then
   signatures that must not hold: another message, another key, a flipped bit, an S
   past the group order, junk keys. ASan. */
#include "../src/ed25519.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void ck(int c, const char *w) { printf("  %s %s\n", c ? "ok  " : "FAIL", w); if (!c) fails++; }

static void unhex(const char *s, unsigned char *out, size_t n) {
	for (size_t i = 0; i < n; i++) { unsigned v; sscanf(s + 2 * i, "%2x", &v); out[i] = (unsigned char)v; }
}
static int sha_is(const void *d, size_t n, const char *want) {
	unsigned char h[64], w[64];
	sha512(d, n, h); unhex(want, w, 64);
	return !memcmp(h, w, 64);
}

/* RFC 8032 7.1, tests 1 to 3 */
static const struct { const char *seed, *pk, *msg, *sig; } V[] = {
	{ "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
	  "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
	  "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b" },
	{ "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
	  "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
	  "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00" },
	{ "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
	  "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
	  "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a" },
};

int main(void) {
	printf("test_keys\n");

	/* ── sha512 ── */
	ck(sha_is("", 0, "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"), "sha512 of nothing");
	ck(sha_is("abc", 3, "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"), "sha512 of abc");
	const char *two = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
	ck(sha_is(two, strlen(two), "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"), "sha512 across two blocks");
	char *ma = malloc(1000000);
	memset(ma, 'a', 1000000);
	ck(sha_is(ma, 1000000, "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b"), "sha512 of a million a's");
	{
		sha512_ctx c; unsigned char h1[64], h2[64];
		sha512_init(&c);
		for (size_t at = 0, k = 1; at < 1000000; at += k, k = k * 7 % 251 + 1)
			sha512_add(&c, ma + at, at + k > 1000000 ? 1000000 - at : k);
		sha512_done(&c, h1);
		sha512(ma, 1000000, h2);
		ck(!memcmp(h1, h2, 64), "sha512 fed in odd pieces is the same hash");
	}
	free(ma);

	/* ── ed25519 ── */
	for (int t = 0; t < 3; t++) {
		unsigned char seed[32], pk[32], want_pk[32], msg[2], sig[64], want_sig[64];
		size_t n = strlen(V[t].msg) / 2;
		unhex(V[t].seed, seed, 32); unhex(V[t].pk, want_pk, 32); unhex(V[t].msg, msg, n); unhex(V[t].sig, want_sig, 64);
		ed25519_public(seed, pk);
		ed25519_sign(sig, msg, n, seed, pk);
		char w[96];
		snprintf(w, sizeof w, "rfc 8032 test %d: public key", t + 1); ck(!memcmp(pk, want_pk, 32), w);
		snprintf(w, sizeof w, "rfc 8032 test %d: signature", t + 1); ck(!memcmp(sig, want_sig, 64), w);
		snprintf(w, sizeof w, "rfc 8032 test %d: it verifies", t + 1); ck(ed25519_verify(sig, msg, n, pk) == 0, w);
	}
	{
		unsigned char s1[32], s2[32], p1[32], p2[32], sig[64], bad[64];
		const char *m = "spelaeum delta";
		memset(s1, 1, 32); memset(s2, 2, 32);
		ed25519_public(s1, p1); ed25519_public(s2, p2);
		ed25519_sign(sig, m, strlen(m), s1, p1);
		ck(ed25519_verify(sig, m, strlen(m), p1) == 0, "a fresh key signs and verifies");
		ck(ed25519_verify(sig, "spelaeum deltA", strlen(m), p1) != 0, "another message does not");
		ck(ed25519_verify(sig, m, strlen(m), p2) != 0, "another key does not");
		int all = 1;
		for (int i = 0; i < 64 * 8; i += 37) { memcpy(bad, sig, 64); bad[i / 8] ^= (unsigned char)(1 << (i % 8)); all &= ed25519_verify(bad, m, strlen(m), p1) != 0; }
		ck(all, "a flipped bit anywhere in the signature does not");
		/* S + L is the same point sum, so only the S < L check refuses it */
		static const unsigned char L[32] = { 0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
		                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x10 };
		memcpy(bad, sig, 64);
		unsigned c = 0;
		for (int i = 0; i < 32; i++) { c += bad[32 + i] + L[i]; bad[32 + i] = (unsigned char)c; c >>= 8; }
		ck(c == 0 && ed25519_verify(bad, m, strlen(m), p1) != 0, "S + L, the same signature written twice, does not");
		int junk = 1;   /* small y: some name no point, the rest someone else's key; none is p1 */
		for (int y = 0; y < 24; y++) { unsigned char np[32] = { (unsigned char)y }; junk &= ed25519_verify(sig, m, strlen(m), np) != 0; }
		ck(junk, "junk keys do not");
		unsigned char s3[32];
		memcpy(s3, s1, 32);
		ed25519_sign(bad, m, strlen(m), s3, p1);
		ck(!memcmp(bad, sig, 64), "signing is deterministic");
	}

	printf(fails ? "test_keys: %d FAILED\n" : "test_keys: all ok\n", fails);
	return fails ? 1 : 0;
}
