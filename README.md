<p align="center">
  <em>Spelaeum: one open model, trained by many machines in rounds. The chain is the model. <code>spel</code>.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-GPL--3.0-ece7da?style=flat-square&labelColor=14130f" alt="GPL-3.0"/>
  <img src="https://img.shields.io/badge/deps-libc%20%2B%20libm%20%2B%20pthread-ece7da?style=flat-square&labelColor=14130f" alt="deps"/>
</p>

<p align="center">
  <sub><a href="https://spelaeum.mithraeum.studio">train in a tab</a> &nbsp;·&nbsp; <a href="https://huggingface.co/datasets/mithraeum/spelaeum">the chain on 🤗</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako">hako</a> (engine and trainer) &nbsp;·&nbsp; <a href="https://mithraeums.github.io">site</a></sub>
</p>

<br>

A spelaeum is the cave a mithraeum is built in. This one is a way for many people to train one open model: the chain's owner provides the data, everyone else provides compute, and nobody adds data of their own.

The first chain runs at [spelaeum.mithraeum.studio](https://spelaeum.mithraeum.studio): press start and the tab trains. It is a 14.65M-weight model from scratch on an open corpus of docs and permissively licensed code, 105M tokens; the blocks, the data and where every piece of the text comes from are on [Hugging Face](https://huggingface.co/datasets/mithraeum/spelaeum).

Each round, trainers train on the slices of the data they are dealt, and the jobs that make the model better go into the next block under the keys that did them. The work a trainer proves is useful work: a training job that made the model better.

## How it works

- **the chain is the model**: block 0 is a starting model, every block after it is a weight delta and its parent's hash; replaying the chain rebuilds the model bit for bit
- **heads link, deltas can go**: a block's hash covers its head, which names the hash of its delta and of the model it makes, so the heads alone link the chain; a trainer that starts late fetches the heads and the current model, never the old deltas
- **a trainer is its key**: an Ed25519 key, made on first use; every delta is signed with it, and each block keeps the signatures, so nobody trains in someone else's name and a replay checks every credit. The name beside it is a label
- **the hash deals the work**: which slice of the data a trainer trains on comes from the parent block's hash and the trainer's key, so nobody picks their own
- **commit, then reveal**: a trainer commits to its delta by hash; the round's seed is made from every commit and picks the text the deltas are checked on, from slices nobody trained that round
- **proof of loss**: a delta counts when it cuts the loss on that text by the difficulty; checking is forward passes, a small part of the cost of training
- **the same on every machine**: the loss is worked out in integers (int8 products, int32 sums, a fixed-point softmax), so a laptop, a phone's browser and a rented GPU box score a delta to the same bit, and a check is a match, not a tolerance; merging is plain float adds and divides, exact everywhere, so a replay is bit for bit on any of them
- **difficulty retargets**: up when most deltas pass, down when most fail
- **merge**: every delta that passes merges into the round's block, unless the best one alone does better
- **credit**: each block records who trained what; `spel credit` adds it up per key, the operations its passed jobs took and a grade from Corax to Pater, each ten times the last
- **expert chains**: on a model with routed experts, each job is dealt a few experts of every layer and trains only those; deltas that touch different experts merge exactly, and a block carries only the experts that moved
- **trunk rounds**: every Nth block of an expert chain trains the shared trunk (attention, routers) instead, with every expert frozen

The whole protocol is in the comment at the top of [`src/chain.h`](src/chain.h).

## Build

`spel` is built on the trainer of the [hako](https://github.com/mithraeums/hako) engine, checked out next to this repo.

```sh
git clone https://github.com/mithraeums/hako
git clone https://github.com/mithraeums/spelaeum
cd spelaeum
make spel        # or: make spel HAKO=/path/to/hako
make test
```

C11, libc and pthread. No Python, no framework, no GPU needed; `make GPU=1` trains on one (OpenCL, loaded at run time), loss checks staying on the CPU in integers.

The same C trains in a browser tab: `make wasm` builds `web/spel.wasm` with [wasi-sdk](https://github.com/WebAssembly/wasi-sdk), `web/wasi.js` gives it a file system, and `web/trainer.js` runs the rounds (the steps of `spel train --url --loop`, with fetch in place of curl). The big files that never change (the data, genesis, blocks) stream to the browser's own storage (OPFS) and are read in place, a window at a time, and a job on the CPU trains its batch a sequence a pass (`SPEL_MICRO`), so a tab needs well under a gigabyte. `make test-web` runs it under node against a real `spel serve`. Where the browser has WebGPU (and JSPI, for the C to wait on a read), a job trains on the GPU: `web/gpu.js` holds the engine's training kernels in WGSL and `web/gpu_web.c` puts them behind the engine's `gpu.h`; checks stay on the CPU. `make test-gpu-web` holds it to the CPU pass under deno.

## Use

```sh
spel init   cave --model start.mlf2 --data corpus.txt   # the owner: a starting model and the data
spel train  cave --name ana                             # a trainer: one training job on the slice its key is dealt
spel seal   cave                                        # close the round: check every delta, write the block
spel verify cave --full                                 # replay from block 0 and re-run every loss check
spel status cave
spel key                                                # this machine's public key (~/.spel/key, made on first use)
spel credit cave [KEY]                                  # every key's passed jobs, operations and grade; KEY for one
spel prune  cave --keep 64                              # cut the deltas off all but the last 64 blocks; heads stay
```

Over a network, one machine is the coordinator and the rest train for it:

```sh
spel serve cave --round 600                                    # hand out the chain, take deltas, seal every 10 minutes
spel train mycopy --name ana --url https://example.org --loop  # fetch, replay, train, send, round after round
spel sync  whole --url https://example.org                     # every block whole, to run spel verify --full on
```

A trainer's copy checks every head it is given (link, seed, difficulty, signatures) and the model's hash before any training on it, and keeps only its last 8 blocks whole. `spel serve` speaks plain HTTP on 127.0.0.1: put TLS in front of it. It keeps 2 GB of its disk free (`SPEL_KEEP_FREE`, in MB) and refuses the deltas of a round that would cut into it. The trainer side needs `curl`. `deploy/` is how spelaeum.mithraeum.studio runs: `spel serve` behind Caddy, the page as static files beside it, and `after.sh` filling the Hugging Face mirror once each block is sealed.

The bytes everyone downloads (the data, the models, the blocks) can come from a mirror instead: the chain directory on any static host, Hugging Face, R2, a farm's own server, or a shared disk as `file://`. The coordinator then sends only the small things and takes the deltas:

```sh
spel serve  cave --files https://huggingface.co/datasets/you/cave/resolve/main --after ./upload.sh
spel mirror cave out --every 8     # lay out cave for a static host in out; prints each path it writes
spel train  mycopy --name ana --url https://example.org --files file:///mnt/cave --loop
```

`--files` on `serve` names the mirror in `/tip`, and trainers take big files from it first; `--files` on a trainer uses its own in its place. `--after PROGRAM` runs `PROGRAM DIR HEIGHT` once a block is sealed: the place to run `spel mirror` and an upload, then write the height reached to `cave/mirrored`, so trainers wait a moment for the mirror rather than all taking the newest block from the coordinator. A mirror is trusted for nothing: blocks link to the coordinator's heads and models are held to a head's hash, and anything amiss is taken from the coordinator. A new copy joins at the mirror's snapshot, which may trail the tip, and takes the blocks after it whole.

A farm runs a relay, so its devices talk to one machine on its own network and the coordinator and the mirror see one client:

```sh
spel relay farmcopy --url https://example.org --bind 0.0.0.0     # a checked copy, answering as the coordinator would
spel train mycopy --name ana --url http://farm-box:7717 --loop   # each device, phone or desktop
```

The relay keeps its copy checked the way a trainer's is, answers `/tip`, `/heads`, the files and `/status.json` from it, and passes each delta on to the coordinator and its answer back.

The seal's work grows with the trainers: every delta's loss is a pass over the check windows. With checks, the trainers do it:

```sh
spel serve cave --round 600 --checks 2 --spot 8 --check-window 60
```

The clock then closes the round instead of sealing it, and trainers waiting for the block check each other's deltas (`/round`, `/checks/KEY`, the round's deltas, `POST /verdict`), each loss signed with their key. The block comes once every delta has its verdicts or the window ends. Two verdicts that agree stand for a delta's loss; one delta in eight, picked by the round's seed, is worked out anyway, and keys that agreed on a wrong loss are barred from checking. A key may check once it has a passed job on the chain, and never its own delta. A check is a forward pass over the windows, about a hundredth of a training job. `spel close` and `spel seal --checks K` do the same by hand.

`cave/tip.mlf2` is the chain's model at its head, and `hakm` runs it as is. `spel` with no arguments lists every option, among them `--experts N` and `--trunk N` for expert chains.

## Where it stands

- one coordinator serves the chain over plain HTTP and seals on a clock; it is trusted to keep the clock and to keep pending deltas to itself until the seal, not for the chain, which anyone replays
- a coordinator that showed pending deltas early would let the last trainer to send influence the seed
- the integer check runs on the CPU and costs about 3.5 times the float pass there
- a pruned block can no longer be re-checked: someone has to keep every block whole for `spel verify --full` to mean anything
- `spel verify --full` re-runs each block's losses, not each delta's: a delta's own loss, and the credit for it, is as good as whoever checked it (the coordinator, or checkers held honest by spot checks)
- measured on a 15M-parameter model trained from scratch on a laptop CPU: three trainers of 100 steps each took perplexity 66.3 → 56.8 → 49.7 over three blocks, sealing cost 42 s against 833 s of training, and a full verify took 14 s (with float checks, before the integer one)

## License

GPL-3.0-or-later
