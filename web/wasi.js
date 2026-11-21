// SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser
// wasi.js — just enough WASI preview 1 for spel.wasm: one directory, preopened as
// /c, held as a Map of path -> file (or a directory marker). The chain's C does its
// own file work (fopen, rename, opendir, truncate) and this answers it. A file is
// { data, size } in memory, or { disk, size }: a sync access handle on a file in the
// browser's own storage (OPFS, workers only), read and written in place, so the
// data (421 MB) never sits in memory, which a phone's tab has no room for.
// No network, no clock beyond Date, no other files. Runs in a page, a worker or node.

const OK = 0, BADF = 8, EXIST = 20, INVAL = 28, ISDIR = 31, NOENT = 44, NOSYS = 52, NOTDIR = 54, NOTEMPTY = 55;
const DIR = 3, REG = 4, CHR = 2;
const ROOT = '/c';

export class Fs {
	constructor() { this.files = new Map(); this.dirs = new Set(['']); }
	norm(p) {
		const out = [];
		for (const s of p.split('/')) {
			if (!s || s === '.') continue;
			if (s === '..') out.pop(); else out.push(s);
		}
		return out.join('/');
	}
	parent(p) { const i = p.lastIndexOf('/'); return i < 0 ? '' : p.slice(0, i); }
	has(p) { return this.files.has(this.norm(p)); }
	read(p) {
		const f = this.files.get(this.norm(p));
		if (!f) return null;
		if (!f.disk) return f.data.subarray(0, f.size);
		const b = new Uint8Array(f.size);
		f.disk.read(b, { at: 0 });
		return b;
	}
	set(p, f) {
		p = this.norm(p);
		for (let d = this.parent(p); d && !this.dirs.has(d); d = this.parent(d)) this.dirs.add(d);
		this.drop(p);
		this.files.set(p, f);
	}
	write(p, bytes) { const data = new Uint8Array(bytes); this.set(p, { data, size: data.length }); }
	putDisk(p, disk) { this.set(p, { disk, size: disk.getSize() }); }
	drop(p) { const f = this.files.get(p); if (f && f.disk) { try { f.disk.close(); } catch {} } }
	remove(p) { p = this.norm(p); this.drop(p); this.files.delete(p); }
	close() { for (const k of this.files.keys()) this.drop(k); this.files.clear(); }
	list(dir) {
		dir = this.norm(dir);
		const pre = dir ? dir + '/' : '', names = new Map();
		for (const k of this.files.keys()) if (k.startsWith(pre) && !k.slice(pre.length).includes('/')) names.set(k.slice(pre.length), REG);
		for (const k of this.dirs) if (k && k.startsWith(pre) && !k.slice(pre.length).includes('/')) names.set(k.slice(pre.length), DIR);
		return [...names];
	}
	bytes() { let n = 0; for (const f of this.files.values()) n += f.size; return n; }
}

import { gpuImports } from './gpu.js';

// run spel.wasm once with args over fs; resolves to { code, out, err }. gpu: openGpu()'s
// device (gpu.js) for a training job to use, or null for the CPU
// tap, when given, sees stderr as it is written: a job's steps while the job runs
export async function run(module, fs, args, env = { HAKM_THREADS: '1' }, gpu = null, tap = null) {
	let mem = null, out = '', err = '';
	const dec = new TextDecoder(), enc = new TextEncoder();
	const fds = new Map();   // fd -> { path, dir, pos, append }
	fds.set(3, { path: '', dir: true, pos: 0 });
	let nextFd = 4;
	const view = () => new DataView(mem.buffer), u8 = () => new Uint8Array(mem.buffer);
	const str = (p, n) => dec.decode(u8().subarray(p, p + n));
	const argv = ['spel', ...args].map((a) => enc.encode(a + '\0'));
	const envv = Object.entries(env).map(([k, v]) => enc.encode(`${k}=${v}\0`));
	const resolve = (fd, p, n) => {
		const d = fds.get(fd);
		if (!d || !d.dir) return null;
		return fs.norm(d.path + '/' + str(p, n));
	};
	const stat = (buf, type, size) => {
		const v = view();
		for (let i = 0; i < 64; i += 8) v.setBigUint64(buf + i, 0n, true);
		v.setUint8(buf + 16, type);
		v.setBigUint64(buf + 24, 1n, true);
		v.setBigUint64(buf + 32, BigInt(size), true);
	};
	class Exit { constructor(code) { this.code = code; } }
	const wasi = {
		args_sizes_get(c, s) { const v = view(); v.setUint32(c, argv.length, true); v.setUint32(s, argv.reduce((a, b) => a + b.length, 0), true); return OK; },
		args_get(ptrs, buf) {
			const v = view();
			argv.forEach((a, i) => { v.setUint32(ptrs + 4 * i, buf, true); u8().set(a, buf); buf += a.length; });
			return OK;
		},
		environ_sizes_get(c, s) { const v = view(); v.setUint32(c, envv.length, true); v.setUint32(s, envv.reduce((a, b) => a + b.length, 0), true); return OK; },
		environ_get(ptrs, buf) {
			const v = view();
			envv.forEach((a, i) => { v.setUint32(ptrs + 4 * i, buf, true); u8().set(a, buf); buf += a.length; });
			return OK;
		},
		clock_time_get(id, prec, t) { view().setBigUint64(t, BigInt(Math.round((id === 0 ? Date.now() : performance.now()) * 1e6)), true); return OK; },
		clock_res_get(id, r) { view().setBigUint64(r, 1000n, true); return OK; },
		random_get(p, n) { crypto.getRandomValues(u8().subarray(p, p + n)); return OK; },
		proc_exit(code) { throw new Exit(code); },
		sched_yield() { return OK; },
		fd_prestat_get(fd, p) {
			if (fd !== 3) return BADF;
			const v = view(); v.setUint8(p, 0); v.setUint32(p + 4, ROOT.length, true);
			return OK;
		},
		fd_prestat_dir_name(fd, p, n) { if (fd !== 3) return BADF; u8().set(enc.encode(ROOT).subarray(0, n), p); return OK; },
		fd_fdstat_get(fd, p) {
			const v = view();
			const d = fds.get(fd);
			if (fd > 2 && !d) return BADF;
			v.setUint8(p, fd <= 2 ? CHR : d.dir ? DIR : REG);
			v.setUint16(p + 2, 0, true);
			v.setBigUint64(p + 8, 0xffffffffffffffffn, true);
			v.setBigUint64(p + 16, 0xffffffffffffffffn, true);
			return OK;
		},
		fd_fdstat_set_flags() { return OK; },
		fd_write(fd, iovs, n, nw) {
			const v = view();
			let total = 0;
			const d = fds.get(fd);
			for (let i = 0; i < n; i++) {
				const p = v.getUint32(iovs + 8 * i, true), l = v.getUint32(iovs + 8 * i + 4, true);
				const b = u8().subarray(p, p + l);
				if (fd === 1) out += dec.decode(b);
				else if (fd === 2) { const t = dec.decode(b); err += t; if (tap) tap(t); }
				else {
					if (!d || d.dir) return BADF;
					const f = fs.files.get(d.path);
					if (d.append) d.pos = f.size;
					const end = d.pos + l;
					if (f.disk) f.disk.write(b, { at: d.pos });
					else {
						if (end > f.data.length) { const g = new Uint8Array(Math.max(end, f.data.length * 2, 4096)); g.set(f.data.subarray(0, f.size)); f.data = g; }
						f.data.set(b, d.pos);
					}
					d.pos = end;
					if (end > f.size) f.size = end;
				}
				total += l;
			}
			v.setUint32(nw, total, true);
			return OK;
		},
		fd_read(fd, iovs, n, nr) { return readAt(fd, iovs, n, nr, null); },
		fd_pread(fd, iovs, n, off, nr) { return readAt(fd, iovs, n, nr, Number(off)); },
		fd_seek(fd, off, whence, np) {
			const d = fds.get(fd);
			if (!d || d.dir) return BADF;
			const f = fs.files.get(d.path), o = Number(off);
			const at = whence === 0 ? o : whence === 1 ? d.pos + o : f.size + o;
			if (at < 0) return INVAL;
			d.pos = at;
			view().setBigUint64(np, BigInt(at), true);
			return OK;
		},
		fd_tell(fd, np) { const d = fds.get(fd); if (!d) return BADF; view().setBigUint64(np, BigInt(d.pos), true); return OK; },
		fd_close(fd) { if (!fds.has(fd) || fd === 3) return fd === 3 ? OK : BADF; fds.delete(fd); return OK; },
		fd_sync() { return OK; }, fd_datasync() { return OK; },
		fd_filestat_get(fd, buf) {
			const d = fds.get(fd);
			if (fd <= 2) { stat(buf, CHR, 0); return OK; }
			if (!d) return BADF;
			stat(buf, d.dir ? DIR : REG, d.dir ? 0 : fs.files.get(d.path).size);
			return OK;
		},
		fd_filestat_set_size(fd, size) {
			const d = fds.get(fd);
			if (!d || d.dir) return BADF;
			const f = fs.files.get(d.path), n = Number(size);
			if (f.disk) f.disk.truncate(n);
			else if (n > f.data.length) { const g = new Uint8Array(n); g.set(f.data.subarray(0, f.size)); f.data = g; }
			else f.data.fill(0, n, f.size);
			f.size = n;
			return OK;
		},
		path_filestat_get(fd, flags, p, n, buf) {
			const path = resolve(fd, p, n);
			if (path === null) return BADF;
			if (fs.dirs.has(path)) { stat(buf, DIR, 0); return OK; }
			const f = fs.files.get(path);
			if (!f) return NOENT;
			stat(buf, REG, f.size);
			return OK;
		},
		path_open(fd, dirflags, p, n, oflags, rb, ri, fdflags, outFd) {
			const path = resolve(fd, p, n);
			if (path === null) return BADF;
			const creat = oflags & 1, directory = oflags & 2, excl = oflags & 4, trunc = oflags & 8;
			if (fs.dirs.has(path)) {
				if (creat && excl) return EXIST;
				fds.set(nextFd, { path, dir: true, pos: 0 });
			} else {
				if (directory) return fs.files.has(path) ? NOTDIR : NOENT;
				let f = fs.files.get(path);
				if (f && creat && excl) return EXIST;
				if (!f) {
					if (!creat) return NOENT;
					if (!fs.dirs.has(fs.parent(path))) return NOENT;
					f = { data: new Uint8Array(0), size: 0 };
					fs.files.set(path, f);
				}
				if (trunc) { if (f.disk) f.disk.truncate(0); f.size = 0; }
				fds.set(nextFd, { path, dir: false, pos: 0, append: !!(fdflags & 1) });
			}
			view().setUint32(outFd, nextFd++, true);
			return OK;
		},
		path_create_directory(fd, p, n) {
			const path = resolve(fd, p, n);
			if (path === null) return BADF;
			if (fs.dirs.has(path) || fs.files.has(path)) return EXIST;
			if (!fs.dirs.has(fs.parent(path))) return NOENT;
			fs.dirs.add(path);
			return OK;
		},
		path_remove_directory(fd, p, n) {
			const path = resolve(fd, p, n);
			if (!fs.dirs.has(path)) return NOENT;
			if (fs.list(path).length) return NOTEMPTY;
			fs.dirs.delete(path);
			return OK;
		},
		path_unlink_file(fd, p, n) {
			const path = resolve(fd, p, n);
			if (path === null) return BADF;
			if (fs.dirs.has(path)) return ISDIR;
			if (!fs.files.has(path)) return NOENT;
			fs.remove(path);
			return OK;
		},
		path_rename(fd, p, n, fd2, q, m) {
			const a = resolve(fd, p, n), b = resolve(fd2, q, m);
			if (a === null || b === null) return BADF;
			const f = fs.files.get(a);
			if (!f) return fs.dirs.has(a) ? NOSYS : NOENT;
			if (fs.dirs.has(b)) return ISDIR;
			if (!fs.dirs.has(fs.parent(b))) return NOENT;
			fs.files.delete(a);
			fs.drop(b);
			fs.files.set(b, f);
			for (const d of fds.values()) if (d.path === a) d.path = b;
			return OK;
		},
		fd_readdir(fd, buf, len, cookie, used) {
			const d = fds.get(fd);
			if (!d || !d.dir) return BADF;
			const ents = [['.', DIR], ['..', DIR], ...fs.list(d.path)];
			const parts = [];
			let total = 0;
			for (let i = Number(cookie); i < ents.length && total < len; i++) {
				const name = enc.encode(ents[i][0]), e = new Uint8Array(24 + name.length), v = new DataView(e.buffer);
				v.setBigUint64(0, BigInt(i + 1), true);
				v.setBigUint64(8, BigInt(i + 1), true);
				v.setUint32(16, name.length, true);
				v.setUint8(20, ents[i][1]);
				e.set(name, 24);
				parts.push(e);
				total += e.length;
			}
			let at = buf;
			const room = buf + len;
			for (const e of parts) { const k = Math.min(e.length, room - at); u8().set(e.subarray(0, k), at); at += k; if (at >= room) break; }
			view().setUint32(used, at - buf, true);
			return OK;
		},
		poll_oneoff() { return NOSYS; },
	};
	function readAt(fd, iovs, n, nr, off) {
		const d = fds.get(fd);
		if (fd === 0) { view().setUint32(nr, 0, true); return OK; }
		if (!d || d.dir) return BADF;
		const f = fs.files.get(d.path), v = view();
		let pos = off === null ? d.pos : off, total = 0;
		for (let i = 0; i < n; i++) {
			const p = v.getUint32(iovs + 8 * i, true), l = v.getUint32(iovs + 8 * i + 4, true);
			const k = Math.max(0, Math.min(l, f.size - pos));
			if (f.disk) { if (k) f.disk.read(u8().subarray(p, p + k), { at: pos }); }   /* straight into the C's memory */
			else u8().set(f.data.subarray(pos, pos + k), p);
			pos += k; total += k;
			if (k < l) break;
		}
		if (off === null) d.pos = pos;
		v.setUint32(nr, total, true);
		return OK;
	}
	const g = gpuImports(gpu, () => mem);
	const imports = { wasi_snapshot_preview1: new Proxy(wasi, { get: (t, k) => t[k] || (() => NOSYS) }), gpu: g };
	const inst = await WebAssembly.instantiate(module, imports);
	mem = inst.exports.memory;
	let code = 0;
	try {   /* with a device the C waits on reads (JSPI), so it runs as a promise */
		if (gpu) await WebAssembly.promising(inst.exports._start)();
		else inst.exports._start();
	} catch (e) { if (e instanceof Exit) code = e.code; else { err += String(e && e.stack || e); code = 134; } }
	g.done();
	if (gpu && gpu.failed) err += `gpu: ${gpu.failed}\n`;
	return { code, out, err };
}
