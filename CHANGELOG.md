# Changelog

All notable changes to Spelaeum (`spel`).

## [v0.0.1] - 2026-11-20

The first cut: one open model, trained by many machines in rounds, the chain is the model. Builds against the hako engine v0.0.7.

### added
- **the chain**: block 0 a starting model, every block after it a weight delta, its parent's hash and the loss it cut, so replaying the chain rebuilds the model bit for bit. `spel init`, `train`, `seal`, `verify`, `status`, `prune`; difficulty set per kind of job; heads that link without their deltas, so a copy joins light
- **every loss checked in integers**: the engine's `iloss`, so a job scores the same to the bit on every machine and in a browser tab, and every check is an exact comparison
- **over a network**: `spel serve` hands out the chain as files, takes deltas and seals on a clock; `spel train --url --loop` trains round after round; `spel sync`; `spel relay` for a farm's own coordinator; `spel mirror` lays the chain out for any static host, Hugging Face among them, so trainers take the big files from there
- **trainer keys** (Ed25519, `spel key`): every delta and verdict signed; `spel credit` and grades, Corax to Pater
- **checks by trainers**: `spel serve --checks K --spot N` closes a round for checks, trainers check each other's deltas while they wait, the seal takes a loss from K verdicts that agree, and works out one in N anyway; a key whose verdict lies is barred
- **a trainer for the browser**: the same C in `web/spel.wasm`, a page that trains in a tab on its CPU or, where there is WebGPU, its GPU, its big files kept in the browser's cache
- the coordinator keeps `SPEL_KEEP_FREE` megabytes of its disk free (2048 unless set)
- `deploy/`: how spelaeum.mithraeum.studio runs
