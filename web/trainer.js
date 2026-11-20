// SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser
// trainer.js — spel train --url --loop for a page: the same steps as net_sync (src/net.c),
// with fetch in place of curl and spel.wasm doing every check and every job. The
// chain's copy lives in memory (wasi.js); the key is a 32-byte seed the page keeps.
// Big files (the data, genesis, blocks, models) come from the mirror the coordinator
// names, or `files` in its place ('' for none), and from the coordinator only when
// the mirror lacks one; everything is checked by spel.wasm wherever it came from.
//
//   const m = new Trainer({ url, module, name, seed, on: (event, data) => ... });
//   await m.loop();   // until m.stop()
//
// A job trains on the GPU when the browser has WebGPU and JSPI (gpu.js), the CPU otherwise;
// either way every check is the engine's integer pass on the CPU, so the page scores a delta to
// the bit as every machine does.

import { Fs, run } from './wasi.js';
import { openGpu } from './gpu.js';

const WHOLE = 8;   // NET_WHOLE: this many blocks behind or less, fetch them whole
const MIRROR_WAIT = 90;   // NET_MIRROR_WAIT: seconds to wait for the mirror to reach the tip

export class Trainer {
	// gpu: train on the GPU when the browser has one (WebGPU and JSPI); false keeps to the CPU
	constructor({ url, files, module, name, seed, on = () => {}, mirrorWait = MIRROR_WAIT, gpu = true }) {
		this.url = url.replace(/\/+$/, '');
		this.files = files;   // undefined: the coordinator's mirror
		this.mirrorWait = mirrorWait;
		this.fromFiles = 0; this.fromUrl = 0;
		this.module = module; this.name = name; this.seed = seed; this.on = on;
		this.fs = new Fs();
		this.fs.dirs.add('pending'); this.fs.dirs.add('blocks');
		this.stopped = false;
		this.useGpu = gpu; this.gpu = undefined;   // undefined: not asked yet; null: none
	}
	// the device a job trains on, opened once; its name, or null for the CPU
	async device() {
		if (this.gpu === undefined) {
			this.gpu = this.useGpu ? await openGpu().catch(() => null) : null;
			this.on('device', { name: this.gpu ? this.gpu.name : null });
		}
		return this.gpu && !this.gpu.failed ? this.gpu.name : null;
	}
	stop() { this.stopped = true; }

	async spel(...args) {
		const gpu = args[0] === 'train' && await this.device() ? this.gpu : null;   // only a job trains
		const r = await run(this.module, this.fs, args, { HAKM_THREADS: '1', SPEL_EVERY: '1' }, gpu, (t) => this.on('stderr', t));
		if (r.err) this.on('log', r.err);
		if (r.code) throw new Error(`spel ${args[0]} failed (${r.code}): ${r.err.trim().split('\n').pop() || ''}`);
		return r.out.trim();
	}
	async get(path, type = 'bytes', base = this.url) {
		const r = await fetch(`${base}/${path}`, { cache: 'no-store' });
		if (!r.ok) { const e = new Error(`${path}: ${r.status}`); e.status = r.status; throw e; }
		return type === 'text' ? r.text() : new Uint8Array(await r.arrayBuffer());
	}
	// a big file: the mirror's (as mpath) when it has it, else the coordinator's (as path). The
	// ones that never change (the data, genesis, its optimizer, blocks) are kept in the browser's
	// cache under the chain's genesis, so a second visit fetches only what is new; the C checks
	// every one against the chain anyway, and a bad one is dropped (forget)
	async getBig(mpath, path) {
		const keep = this.cache && (path === 'ids.bin' || path === 'genesis.mlf2' || path === 'genesis.opt' || path.startsWith('blocks/'));
		if (keep) {
			try { const hit = await (await caches.open(this.cache)).match(`/${path}`); if (hit) { this.on('fetch', { name: path, from: 'cache' }); return new Uint8Array(await hit.arrayBuffer()); } } catch {}
		}
		let b = null;
		if (this.mirror) {
			try { b = await this.get(mpath, 'bytes', this.mirror); this.fromFiles++; this.on('fetch', { name: path, from: 'mirror' }); } catch {}
		}
		if (!b) { b = await this.get(path); this.fromUrl++; this.on('fetch', { name: path, from: 'coordinator' }); }
		if (keep) { try { await (await caches.open(this.cache)).put(`/${path}`, new Response(b)); } catch {} }
		return b;
	}
	async forget() { if (this.cache) { try { await caches.delete(this.cache); } catch {} } }
	mirrorBlock(h) { return `blocks/${String(Math.floor(h / 1000)).padStart(3, '0')}/${String(h).padStart(6, '0')}.blk`; }
	async mirrorTip() {
		const [h, hash] = (await this.get('tip', 'text', this.mirror)).trim().split(' ');
		if (!/^\d+$/.test(h) || !/^[0-9a-f]{64}$/.test(hash || '')) throw new Error('not a mirror');
		return Number(h);
	}
	async putHeads(from, upto) {
		while (from <= upto) {
			this.fs.write('heads.part', await this.get(`heads/${from}`));
			from += Number(await this.spel('heads', '/c', String(from), '/c/heads.part', String(upto)));
			this.fs.remove('heads.part');
		}
	}
	async tip() {
		const t = {};
		for (const line of (await this.get('tip', 'text')).split('\n')) {
			const [k, v] = line.split(' ');
			if (k) t[k] = k === 'hash' || k === 'files' || k === 'phase' ? v : Number(v);
		}
		if (!('height' in t) || !t.hash) throw new Error('not a coordinator');
		return t;
	}
	localTip() {
		const b = this.fs.read('tip');
		if (!b) return null;
		const [h, hash] = new TextDecoder().decode(b).trim().split(' ');
		return { height: Number(h), hash };
	}
	block(h) { return `blocks/${String(h).padStart(6, '0')}.blk`; }

	// bring the copy up to the coordinator's tip; resolves to that tip
	async sync() {
		let t = await this.tip();
		const named = /^https?:\/\/[^\s"']+$/.test(t.files || '') ? t.files : '';
		this.mirror = (this.files !== undefined ? this.files : named).replace(/\/+$/, '');
		if (!this.fs.read('chain.cfg')) { this.on('fetch', { name: 'chain.cfg', from: 'coordinator' }); this.fs.write('chain.cfg', await this.get('chain.cfg')); }
		const g = /^genesis ([0-9a-f]{64})$/m.exec(new TextDecoder().decode(this.fs.read('chain.cfg')));
		this.cache = g && typeof caches !== 'undefined' ? `spelaeum-${g[1]}` : null;
		for (const f of ['genesis.mlf2', 'ids.bin']) if (!this.fs.read(f)) this.fs.write(f, await this.getBig(f, f));
		if (!this.fs.read('genesis.opt')) { try { this.fs.write('genesis.opt', await this.getBig('genesis.opt', 'genesis.opt')); } catch (e) { if (e.status !== 404) throw e; } }
		try { await this.spel('start', '/c'); } catch (e) { await this.forget(); throw e; }
		for (let tries = 0; ; tries++) {
			if (tries) t = await this.tip();
			let mine = this.localTip();
			if (mine.height > t.height) throw new Error('this copy is past the coordinator: not the same chain');
			// a coordinator that says how far its mirror has got: wait a while for it
			for (const until = Date.now() + this.mirrorWait * 1000; this.mirror && 'mirrored' in t && t.mirrored < t.height && mine.height < t.height && Date.now() < until;) {
				this.on('log', `waiting for the mirror to reach block ${t.height}\n`);
				await sleep(Math.min(5000, Math.max(1000, until - Date.now())));
				try { t = await this.tip(); } catch {}
			}
			if (t.height - mine.height <= WHOLE) {
				for (let h = mine.height + 1; h <= t.height; h++) this.fs.write(this.block(h), await this.getBig(this.mirrorBlock(h), this.block(h)));
				try { await this.spel('advance', '/c'); return this.check(t); } catch { this.on('log', 'taking the heads and the model instead\n'); }
			}
			// a mirror's snapshot between here and the tip: heads up to it, its model, the blocks after it whole
			if (this.mirror) {
				let s = -1;
				try { s = await this.mirrorTip(); } catch {}
				if (s > mine.height && s <= t.height) {
					try {
						await this.putHeads(mine.height + 1, s);
						this.fs.write('tip.new', await this.get('tip.mlf2', 'bytes', this.mirror));
						this.fromFiles++; this.on('fetch', { name: 'tip.mlf2', from: 'mirror' });
						await this.spel('join', '/c', 'tip.new');
						for (let h = s + 1; h <= t.height; h++) this.fs.write(this.block(h), await this.getBig(this.mirrorBlock(h), this.block(h)));
						if (s < t.height) await this.spel('advance', '/c');
						return this.check(t);
					} catch { this.fs.remove('tip.new'); this.on('log', 'the mirror\'s snapshot did not do: taking the coordinator\'s\n'); }
					mine = this.localTip();
				}
			}
			await this.putHeads(mine.height + 1, t.height);
			this.on('fetch', { name: 'tip.mlf2', from: 'coordinator' });
			this.fs.write('tip.new', await this.get('tip.mlf2'));
			this.fromUrl++;
			try { await this.spel('join', '/c', 'tip.new'); return this.check(t); } catch (e) {
				this.fs.remove('tip.new');
				if (tries === 2) throw e;   // sealed between the heads and the model: try again
			}
		}
	}
	async check(t) {
		const mine = this.localTip();
		if (mine.height !== t.height || mine.hash !== t.hash) throw new Error('the blocks replay to another tip than the coordinator names');
		await this.spel('prune', '/c', String(WHOLE));
		return t;
	}

	async trainOnce() {
		const t = await this.sync();
		this.on('job', { height: t.height });
		const hex = await this.spel('train', '/c', this.name, this.seed);
		const path = `pending/${hex}.dlt`, body = this.fs.read(path);
		const r = await fetch(`${this.url}/submit`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body });
		const said = (await r.text()).trim();
		this.fs.remove(path);
		this.on(r.ok ? 'sent' : 'refused', { height: t.height, hex, said });
		return t;
	}

	async publicKey() { return this.pk || (this.pk = await this.spel('key', this.seed)); }

	// a closed round: check what the coordinator hands this key, on the copy (the round's
	// parent), each loss signed and sent as a verdict; resolves to how many it sent
	async checkRound() {
		const pk = await this.publicKey();
		let list;
		try {
			this.fs.write('round.remote', await this.get('round'));
			list = (await this.get(`checks/${pk}`, 'text')).split('\n').filter((x) => /^[0-9a-f]{64}$/.test(x));
		} catch { this.fs.remove('round.remote'); return 0; }
		let sent = 0;
		for (const c of list) {
			try {
				this.fs.write('check.dlt', await this.get(`pending/${c}.dlt`));
				const hex = await this.spel('check', '/c', '/c/round.remote', '/c/check.dlt', this.seed);
				const body = Uint8Array.from(hex.match(/../g).map((h) => parseInt(h, 16)));
				const r = await fetch(`${this.url}/verdict`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body });
				const said = (await r.text()).trim();
				if (r.ok) { sent++; this.on('checked', { commit: c }); } else this.on('log', `a check was not taken: ${said}\n`);
			} catch (e) { this.on('log', `a check failed: ${e.message || e}\n`); }
			this.fs.remove('check.dlt');
		}
		this.fs.remove('round.remote');
		return sent;
	}

	// this key's credit on the copy (heads alone): { passed, merged, refused, ops, grade, next }
	async credit(pk) { return JSON.parse(await this.spel('credit', '/c', pk)); }

	async loop() {
		while (!this.stopped) {
			let t;
			try { t = await this.trainOnce(); } catch (e) {
				this.on('error', String(e.message || e));
				await sleep(30000);
				continue;
			}
			this.on('wait', { seal: t.seal, now: t.now });
			let checked = false;
			for (let now = t; !this.stopped && now.height === t.height;) {   // until the round seals, checking others' once it closes
				const wait = Math.max(2, Math.min(30, (now.seal || 0) - (now.now || 0)));
				await sleep(wait * 1000);
				try { now = await this.tip(); } catch { await sleep(10000); continue; }
				if (now.phase === 'checking' && now.height === t.height && !checked) { checked = true; await this.checkRound(); }
			}
		}
	}
}

export function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
export function newSeed() { const b = crypto.getRandomValues(new Uint8Array(32)); return [...b].map((x) => x.toString(16).padStart(2, '0')).join(''); }
// the key as a file, the same text as spel's ~/.spel/key: a page's key and a terminal's
// are one key, so a tab's credit survives cleared site data and moves to a desktop
export function keyText(seed) { return `spelaeum key\n${seed}\n`; }
export function seedOf(text) {
	const m = /^spelaeum key\n([0-9a-f]{64})\s*$/.exec(String(text));
	if (!m) throw new Error('not a spelaeum key');
	return m[1];
}
