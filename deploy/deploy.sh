#!/bin/sh
# deploy.sh: put the spelaeum coordinator on its box, the one the hako relay runs on.
# You run this from your desk with your key; spel is built on the box from the sources
# sent over ssh (this clone and the hako engine beside it), so nothing binary is copied
# but the page's spel.wasm.
#   ./deploy.sh              build spel, install the service, the Caddy site and the page; (re)start; check
#   ./deploy.sh mirror       once, from this machine: the chain's first files to the Hugging Face
#                            dataset (made private if it is new; make it public to go live)
#   ./deploy.sh chain        once: the chain itself (CAVE) onto the box; refused if the box has one
#   ./deploy.sh hf           the box's Hugging Face token (write access to the dataset, nothing
#                            else), asked for here and sent over ssh stdin only
#   ./deploy.sh status       the service and its last log lines
#   ./deploy.sh box          what the box has: cores, memory, disk
# KEY=path, HOST=user@host, CAVE=dir and REPO=org/name override the defaults.
set -eu
KEY=${KEY:-$HOME/.ssh/mithraeum-relay.pem}
HOST=${HOST:-admin@relay.mithraeum.studio}
CAVE=${CAVE:-$HOME/.hako/dl/spelaeum/cave}
REPO=${REPO:-mithraeum/spelaeum}
SITE=spelaeum.mithraeum.studio
here=$(cd "$(dirname "$0")/.." && pwd)
hako=${HAKO:-$here/../hako}
on_box() { ssh -i "$KEY" -o IdentitiesOnly=yes "$HOST" "$@"; }
need_key() { [ -r "$KEY" ] || { echo "no key at $KEY (KEY=path ./deploy.sh)"; exit 1; }; }

case "${1:-deploy}" in
box) need_key; on_box 'nproc; free -m | sed -n 2p; df -h / | tail -1; uname -m; . /etc/os-release; echo "$PRETTY_NAME"'; exit ;;
status) need_key; on_box "systemctl --no-pager status spelaeum | head -6; sudo journalctl -u spelaeum -n 30 --no-pager"; exit ;;
hf)
	need_key
	stty -echo; printf 'Hugging Face token, write access to %s only (not shown): ' "$REPO"; read -r tok; stty echo; echo
	printf '%s\n' "$tok" | on_box "sudo -u spelaeum sh -c 'umask 077; mkdir -p /var/lib/spelaeum/hf; cat > /var/lib/spelaeum/hf/token'"
	on_box "sudo -u spelaeum env HF_HOME=/var/lib/spelaeum/hf /var/lib/spelaeum/venv/bin/hf auth whoami"; exit ;;
mirror)
	# the data, genesis and the first snapshot from here: the box would otherwise send each
	# new trainer half a gigabyte before its first seal. The box's after.sh takes it from there.
	[ -f "$CAVE/chain.cfg" ] || { echo "no chain at $CAVE"; exit 1; }
	out=$(dirname "$CAVE")/mirror
	"$here/spel" mirror "$CAVE" "$out" >/dev/null
	# the chain's own papers sit beside it: NOTICE.md and licenses/ (where the text is from), README.md (the card)
	data=$(dirname "$CAVE")
	for f in NOTICE.md README.md licenses; do [ -e "$data/$f" ] || { echo "no $data/$f"; exit 1; }; done
	cp "$data/NOTICE.md" "$data/README.md" "$out/"; rm -rf "$out/licenses"; cp -R "$data/licenses" "$out/licenses"
	hf upload "$REPO" "$out" . --repo-type dataset --private --commit-message "the chain at height 0: data, genesis, snapshot"
	echo "mirror: https://huggingface.co/datasets/$REPO (private until you make it public)"; exit ;;
chain)
	need_key
	[ -f "$CAVE/chain.cfg" ] || { echo "no chain at $CAVE"; exit 1; }
	"$here/spel" verify "$CAVE"
	on_box 'sudo test ! -e /var/lib/spelaeum/cave' || { echo "the box already has a chain: not touching it"; exit 1; }
	echo 0 > "$CAVE/mirrored"
	# the chain whole; the mirror's copies of the big files only as names, since they are on Hugging Face already
	tar -C "$(dirname "$CAVE")" -cf - "$(basename "$CAVE")" | on_box 'set -e
sudo mkdir -p /var/lib/spelaeum && sudo tar -C /var/lib/spelaeum -xf -
sudo mkdir -p /var/lib/spelaeum/mirror/blocks
for f in chain.cfg genesis.mlf2 ids.bin tip tip.mlf2; do sudo touch /var/lib/spelaeum/mirror/$f; done
sudo chown -R spelaeum:spelaeum /var/lib/spelaeum'
	echo "chain: on the box, $(du -sh "$CAVE" | cut -f1)"; exit ;;
deploy) need_key ;;
*) echo "usage: $0 [deploy | mirror | chain | hf | status | box]"; exit 2 ;;
esac

[ -f "$hako/src/train.c" ] || { echo "no hako engine at $hako (HAKO=path ./deploy.sh)"; exit 1; }
(cd "$here" && make wasm >/dev/null)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/web" "$stage/hako"
cp -R "$here/Makefile" "$here/cli" "$here/src" "$here/deploy" "$stage/"
cp "$here/web/index.html" "$here/web/worker.js" "$here/web/trainer.js" "$here/web/wasi.js" "$here/web/gpu.js" "$here/web/spel.wasm" "$here/web/icon.svg" "$stage/web/"
cp -R "$hako/src" "$hako/include" "$stage/hako/"
rm -f "$stage"/src/*.o "$stage"/hako/src/*.o
tar -C "$stage" -czf - . | on_box 'rm -rf /tmp/spelaeum && mkdir -p /tmp/spelaeum && tar -C /tmp/spelaeum -xzf -'
on_box 'set -e
cd /tmp/spelaeum
command -v cc >/dev/null || { sudo apt-get update -q && sudo apt-get install -y -q build-essential; }
command -v curl >/dev/null || sudo apt-get install -y -q curl
python3 -c "import venv, ensurepip" 2>/dev/null || sudo apt-get install -y -q python3-venv
make spel HAKO=hako
id spelaeum >/dev/null 2>&1 || sudo useradd --system --no-create-home --shell /usr/sbin/nologin spelaeum
sudo install -d -o spelaeum -g spelaeum -m 750 /var/lib/spelaeum
[ -x /var/lib/spelaeum/venv/bin/hf ] || { sudo -u spelaeum python3 -m venv /var/lib/spelaeum/venv && sudo -u spelaeum /var/lib/spelaeum/venv/bin/pip install -q huggingface_hub; }
sudo install -m 755 spel /usr/local/bin/spel
sudo install -d /usr/local/lib/spelaeum
sudo install -m 755 deploy/after.sh /usr/local/lib/spelaeum/after.sh
sudo install -m 644 deploy/spelaeum.service /etc/systemd/system/spelaeum.service
sudo install -d /var/www/spelaeum /etc/caddy/sites
sudo install -m 644 web/index.html web/worker.js web/trainer.js web/wasi.js web/gpu.js web/spel.wasm web/icon.svg /var/www/spelaeum/
sudo install -m 644 deploy/spelaeum.caddy /etc/caddy/sites/spelaeum.caddy
# the relay owns /etc/caddy/Caddyfile; it imports the sites folder (hako-studio relay/deploy/Caddyfile)
grep -q "^import /etc/caddy/sites/\*.caddy" /etc/caddy/Caddyfile || sudo sed -i "1i import /etc/caddy/sites/*.caddy" /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile >/dev/null
sudo systemctl daemon-reload
if sudo test -f /var/lib/spelaeum/cave/chain.cfg; then sudo systemctl enable spelaeum >/dev/null 2>&1; sudo systemctl restart spelaeum; else echo "no chain on the box yet: ./deploy.sh chain, then deploy again"; fi
sudo systemctl reload caddy
cd / && rm -rf /tmp/spelaeum'
sleep 2
curl -fsS "https://$SITE/tip" >/dev/null && echo "spelaeum: live at https://$SITE" \
	|| { echo "spelaeum: not answering yet (DNS for $SITE, or ./deploy.sh status)"; exit 1; }
