#!/bin/sh
# test_after.sh — deploy/after.sh, the coordinator's hook that mirrors each sealed block to
# Hugging Face, on a real chain with a stand-in for the hf CLI: a block whose upload failed
# is sent by the next run, the mirror's height counts only what got up, and a sent block is
# left as an empty name. Run by make test; needs the engine's hakm-train.
set -e
cd "$(dirname "$0")/.."
HAKO=${HAKO:-../hako}
B=build/test_after-$$
mkdir -p "$B"
trap 'rm -rf "$B"' EXIT INT TERM
fails=0
ck() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; fails=$((fails + 1)); fi; }   # ck WHAT CONDITION

i=0; while [ $i -lt 400 ]; do
	printf 'the box sat on the shelf and the box was old. the old box held a key, and the key opened the door at the end of the hall.\n'
	printf 'she took the key from the box and walked to the door. the door was locked, the key turned, and the hall was quiet again.\n'
	i=$((i + 1)); done > "$B/data.txt"
"$HAKO/hakm-train" --data "$B/data.txt" --out "$B/g.mlf2" --vocab 320 --dim 32 --layers 2 --heads 2 --ctx 32 --steps 20 --batch 4 \
	--eval-every 1000 --save-every 1000 --log-every 1000 --threads 2 >/dev/null 2>&1
./spel init "$B/cave" --model "$B/g.mlf2" --opt "$B/g.mlf2.opt" --data "$B/data.txt" --shards 8 --steps 10 --batch 4 --eval 8 --stripe 64 2>/dev/null
./spel train "$B/cave" --name t --key "$B/t.key" >/dev/null 2>&1
./spel seal "$B/cave" >/dev/null 2>&1
h=$(./spel status "$B/cave" 2>/dev/null | sed -n 's/.*height \([0-9][0-9]*\).*/\1/p' | head -1)
ck "a chain with one block sealed" '[ "$h" = 1 ]'

# the box's mirror as deploy.sh chain leaves it: what is on Hugging Face already, as empty names
mkdir -p "$B/mirror/blocks"
for f in chain.cfg genesis.mlf2 genesis.opt ids.bin tip tip.mlf2; do : > "$B/mirror/$f"; done
# hf: refuses while $B/down exists, else notes the path it was sent
cat > "$B/hf" <<EOF
#!/bin/sh
[ ! -e "$B/down" ] || exit 1
echo "\$4" >> "$B/sent"
EOF
chmod +x "$B/hf"
run() { SPEL_MIRROR_OUT="$B/mirror" SPEL_HF="$B/hf" SPEL_BIN=./spel sh deploy/after.sh "$B/cave" 1 >/dev/null 2>&1; }
blk=blocks/000/000001.blk

: > "$B/down"
run && r=0 || r=$?
ck "Hugging Face down: the hook fails" '[ "$r" != 0 ]'
ck "  and takes the block back out of the mirror" '[ ! -e "$B/mirror/$blk" ]'
ck "  and the mirror's height is not moved" '[ ! -e "$B/cave/mirrored" ]'

rm -f "$B/down"
run && r=0 || r=$?
ck "Hugging Face up again: the next run goes through" '[ "$r" = 0 ]'
ck "  and sends the block it could not before" 'grep -qx "$blk" "$B/sent" 2>/dev/null'
ck "  and a snapshot, the model and its tip" 'grep -qx tip "$B/sent" 2>/dev/null && grep -qx tip.mlf2 "$B/sent"'
ck "  the sent block left as an empty name" '[ -e "$B/mirror/$blk" ] && [ ! -s "$B/mirror/$blk" ]'
ck "  the mirror's height is 1" '[ "$(cat "$B/cave/mirrored" 2>/dev/null)" = 1 ]'

: > "$B/sent"
run && r=0 || r=$?
ck "a run with nothing new sends no block again" '[ "$r" = 0 ] && ! grep -qx "$blk" "$B/sent"'

[ $fails = 0 ] && echo "test_after: all ok" || { echo "test_after: $fails FAILED"; exit 1; }
