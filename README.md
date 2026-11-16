<p align="center">
  <em>Spelaeum: block mining applied to training. The chain is the model. <code>spel</code>.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-GPL--3.0-ece7da?style=flat-square&labelColor=14130f" alt="GPL-3.0"/>
  <img src="https://img.shields.io/badge/deps-libc%20%2B%20libm%20%2B%20pthread-ece7da?style=flat-square&labelColor=14130f" alt="deps"/>
  <img src="https://img.shields.io/badge/coin-none-ece7da?style=flat-square&labelColor=14130f" alt="no coin"/>
</p>

<p align="center">
  <sub><a href="https://mithraeums.github.io">site</a> &nbsp;·&nbsp; <a href="https://github.com/mithraeums/hako">hako</a> (engine and trainer) &nbsp;·&nbsp; <a href="https://github.com/mithraeums">org</a></sub>
</p>

<br>

A spelaeum is the cave a mithraeum is built in. This one is a way for many people to train one open model: the chain's owner provides the data, everyone else provides compute, and nobody adds data of their own.

It borrows the mechanics of block mining and nothing else. There is no coin, no token and nothing to buy. The work a miner proves is useful work: a training job that made the model better.

## How it works

- **the chain is the model**: block 0 is a starting model, every block after it is a weight delta and its parent's hash; replaying the chain rebuilds the model bit for bit
- **the hash deals the work**: which slice of the data a miner trains on comes from the parent block's hash and the miner's name, so nobody picks their own
- **commit, then reveal**: a miner commits to its delta by hash; the round's seed is made from every commit and picks the text the deltas are checked on, from slices nobody mined that round
- **proof of loss**: a delta counts when it cuts the loss on that text by the difficulty; checking is forward passes, a small part of the cost of mining
- **difficulty retargets**: up when most deltas pass, down when most fail
- **merge**: every delta that passes merges into the round's block, unless the best one alone does better
- **credit**: each block records who mined what, and that record is the only reward
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

C11, libc and pthread. No Python, no framework, no GPU needed.

## Use

```sh
spel init   cave --model start.mlf2 --data corpus.txt   # the owner: a starting model and the data
spel mine   cave --miner ana                            # a miner: one training job on the slice ana is dealt
spel seal   cave                                        # close the round: check every delta, write the block
spel verify cave --full                                 # replay from block 0 and re-run every loss check
spel status cave
```

`cave/tip.mlf2` is the chain's model at its head, and `hakm` runs it as is. `spel` with no arguments lists every option, among them `--experts N` and `--trunk N` for expert chains.

## Where it stands

- a chain is a directory on one machine; there is no network layer, so miners share it by sharing the directory
- miner names are not authenticated
- the last miner to commit in a round can influence the seed
- replay is bit for bit on one kind of hardware; floating point can differ across kinds
- measured on a 15M-parameter model trained from scratch on a laptop CPU: three miners of 100 steps each took perplexity 66.3 → 56.8 → 49.7 over three blocks, sealing cost 42 s against 833 s of mining, and a full verify took 14 s

## License

GPL-3.0-or-later
