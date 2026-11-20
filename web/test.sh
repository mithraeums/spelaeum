#!/bin/sh
# test.sh — the page's trainer (spel.wasm + wasi.js + trainer.js) under node against a real
# spel serve on localhost: a tiny genesis from the engine's hakm-train, a chain, the
# coordinator, then web/test.mjs. Run by make test-web; needs node.
set -e
cd "$(dirname "$0")/.."
HAKO=${HAKO:-../hako}
B=build/test_web-$$
mkdir -p "$B"
trap 'kill $SRV 2>/dev/null; rm -rf "$B"' EXIT INT TERM
i=0; while [ $i -lt 400 ]; do
	printf 'the box sat on the shelf and the box was old. the old box held a key, and the key opened the door at the end of the hall.\n'
	printf 'she took the key from the box and walked to the door. the door was locked, the key turned, and the hall was quiet again.\n'
	i=$((i+1)); done > "$B/data.txt"
"$HAKO/hakm-train" --data "$B/data.txt" --out "$B/g.mlf2" --vocab 320 --dim 32 --layers 2 --heads 2 --ctx 32 --steps 20 --batch 4 \
	--eval-every 1000 --save-every 1000 --log-every 1000 --threads 2 >/dev/null 2>&1
./spel init "$B/cave" --model "$B/g.mlf2" --opt "$B/g.mlf2.opt" --data "$B/data.txt" --shards 8 --steps 10 --batch 4 --eval 8 --stripe 64 2>/dev/null
PORT=$((20000 + $$ % 20000))
./spel serve "$B/cave" --port $PORT --round 3600 2>/dev/null & SRV=$!
sleep 1
node web/test.mjs "http://127.0.0.1:$PORT" "$B/cave" ./spel
