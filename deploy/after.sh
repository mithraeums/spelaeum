#!/bin/sh
# after.sh DIR HEIGHT: spel serve runs this once a block is sealed (--after), one at a time.
# Lays the chain out for the mirror (spel mirror), uploads what is new to the Hugging Face
# dataset, then says how far the mirror has got (DIR/mirrored), so trainers take the big
# files from there. Uploaded blocks are left as empty files here (spel mirror only asks
# whether one exists), and the chain's own old blocks are pruned: Hugging Face keeps them whole.
set -eu
dir=$1 height=$2
repo=${SPEL_MIRROR_REPO:-mithraeum/spelaeum}
out=/var/lib/spelaeum/mirror
hf=/var/lib/spelaeum/venv/bin/hf
list=$(mktemp)
trap 'rm -f "$list"' EXIT
/usr/local/bin/spel mirror "$dir" "$out" --every 8 > "$list"
while read -r path; do
	[ -n "$path" ] || continue
	case "$path" in tip|tip.mlf2) continue ;; esac   # the snapshot goes last, model before the tip that names it
	"$hf" upload "$repo" "$out/$path" "$path" --repo-type dataset --commit-message "block $height: $path" --format quiet >/dev/null
	case "$path" in blocks/*) : > "$out/$path" ;; esac
done < "$list"
if grep -qx tip "$list"; then
	"$hf" upload "$repo" "$out/tip.mlf2" tip.mlf2 --repo-type dataset --commit-message "block $height: snapshot" --format quiet >/dev/null
	"$hf" upload "$repo" "$out/tip" tip --repo-type dataset --commit-message "block $height: snapshot tip" --format quiet >/dev/null
fi
echo "$height" > "$dir/mirrored.part" && mv "$dir/mirrored.part" "$dir/mirrored"
/usr/local/bin/spel prune "$dir" --keep 64 >/dev/null
