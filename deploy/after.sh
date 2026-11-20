#!/bin/sh
# after.sh DIR HEIGHT: spel serve runs this once a block is sealed (--after), one at a time.
# Lays the chain out for the mirror (spel mirror), uploads what is new to the Hugging Face
# dataset, then says how far the mirror has got (DIR/mirrored), so trainers take the big
# files from there. Uploaded blocks are left as empty files here (spel mirror only asks
# whether one exists), and the chain's own old blocks are pruned: Hugging Face keeps them whole.
set -eu
dir=$1 height=$2
repo=${SPEL_MIRROR_REPO:-mithraeum/spelaeum}
out=${SPEL_MIRROR_OUT:-/var/lib/spelaeum/mirror}   # these three are only moved by tests/test_after.sh
hf=${SPEL_HF:-/var/lib/spelaeum/venv/bin/hf}
spel=${SPEL_BIN:-/usr/local/bin/spel}
list=$(mktemp) sent=$(mktemp)
trap 'rm -f "$list" "$sent"' EXIT
# spel mirror takes whatever is in $out as mirrored already, so a path that did not get up
# must not stay there: on a failed upload every path of this run not sent yet is taken
# back, the mirror's height is left where it was, and the next block's run sends them again
up() {
	if "$hf" upload "$repo" "$out/$1" "$1" --repo-type dataset --commit-message "$2" --format quiet </dev/null >/dev/null; then
		echo "$1" >> "$sent"
		return 0
	fi
	while read -r p; do
		if [ -n "$p" ] && ! grep -qxF "$p" "$sent"; then rm -f "$out/$p"; fi
	done < "$list"
	echo "after: $1 did not reach $repo; the next block's run sends it again" >&2
	exit 1
}
"$spel" mirror "$dir" "$out" --every 8 > "$list"
while read -r path; do
	[ -n "$path" ] || continue
	case "$path" in tip|tip.mlf2) continue ;; esac   # the snapshot goes last, model before the tip that names it
	up "$path" "block $height: $path"
	case "$path" in blocks/*) : > "$out/$path" ;; esac
done < "$list"
if grep -qx tip "$list"; then
	up tip.mlf2 "block $height: snapshot"
	up tip "block $height: snapshot tip"
fi
echo "$height" > "$dir/mirrored.part" && mv "$dir/mirrored.part" "$dir/mirrored"
"$spel" prune "$dir" --keep 64 >/dev/null
