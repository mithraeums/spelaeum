// SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser
// test.mjs — the page's trainer under node against a real coordinator: node web/test.mjs URL DIR
// URL is a running spel serve; DIR is that coordinator's chain directory, sealed here by
// the test itself with spel seal so it need not wait for the clock. Checks: a first sync
// from nothing, a trained delta the coordinator takes and the seal merges or rates, a
// second round from the copy's own tip, the same job trained twice is the same delta,
// a delta signed by another key is refused, a late tab's credit from heads alone is the
// coordinator's, a page's key file is the terminal's key, and a tab joins from a mirror
// (spel mirror's layout on a plain static server) with only what it lacks from the coordinator,
// and a tab checks a closed round's delta, its verdict the native seal's loss to the bit.
import { readFile } from 'node:fs/promises';
import { execFileSync, spawnSync } from 'node:child_process';
import { Trainer, newSeed, keyText, seedOf } from './trainer.js';
import { writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, join } from 'node:path';
import { run } from './wasi.js';

const [url, dir, spel = './spel'] = process.argv.slice(2);
let fails = 0;
const ck = (c, w) => { console.log(`  ${c ? 'ok  ' : 'FAIL'} ${w}`); if (!c) fails++; };
const module = await WebAssembly.compile(await readFile(new URL('./spel.wasm', import.meta.url)));
const seal = () => execFileSync(spel, ['seal', dir], { stdio: ['ignore', 'ignore', 'pipe'] }).toString();
const events = [];
const seed = newSeed();
const m = new Trainer({ url, module, name: 'tab', seed, on: (e, d) => events.push([e, d]) });

console.log('test_web');
const pk = (await run(module, m.fs, ['key', seed])).out.trim();
ck(/^[0-9a-f]{64}$/.test(pk), 'a seed makes a public key');
const t0 = await m.trainOnce();
ck(events.some(([e]) => e === 'sent'), `a first round from nothing: synced to ${t0.height}, trained, sent`);
const sent = events.find(([e]) => e === 'sent')?.[1];
let out = '';
try { out = seal(); } catch (e) { out = String(e.stderr || ''); }
const status = JSON.parse(execFileSync('curl', ['-s', `${url}/status.json`]).toString());
const mine = status.blocks[0]?.trainers.find((x) => x.key === pk);
ck(status.height === t0.height + 1 && mine && mine.name === 'tab', `the seal rates it under its key (${mine?.state}, block ${status.height})`);
const before = m.fs.bytes();
const t1 = await m.trainOnce();
ck(t1.height === t0.height + 1 && !events.some(([e, d]) => e === 'log' && /instead/.test(d)), 'a second round advances from the copy\'s own tip');
const second = events.filter(([e]) => e === 'sent').pop()?.[1];
ck(second && second.hex !== sent.hex, 'a new block, a new delta');
// the same job again: same key, same tip, same optimizer state -> the same delta, refused as a second one
m.fs.remove('trainers/tab.opt');
const again = await m.spel('train', '/c', 'tab', seed);
const r = await fetch(`${url}/submit`, { method: 'POST', body: m.fs.read(`pending/${again}.dlt`) });
ck(!r.ok && /already/.test(await r.text()), 'a second delta from one key in a round is refused');
m.fs.remove(`pending/${again}.dlt`);
const other = new Trainer({ url, module, name: 'tab', seed: newSeed(), on: () => {} });
other.fs = m.fs;
const forged = await other.spel('train', '/c', 'tab2', newSeed());
const fb = m.fs.read(`pending/${forged}.dlt`).slice();
fb[fb.length - 3] ^= 1;
const r2 = await fetch(`${url}/submit`, { method: 'POST', body: fb });
ck(!r2.ok && /signature/.test(await r2.text()), 'a delta whose signature is not its key\'s is refused');
ck(m.fs.bytes() < before * 3, `the copy stays small: ${(m.fs.bytes() / 1e6).toFixed(1)} MB in memory`);
// ten more blocks, then a tab that opens late: heads and the model, no old delta
try { seal(); } catch {}
for (let i = 0; i < 10; i++) { await m.trainOnce(); try { seal(); } catch {} }
const late = [], tab2 = new Trainer({ url, module, name: 'late', seed: newSeed(), on: (e, d) => late.push([e, d]) });
const lt = await tab2.sync();
const tipNow = JSON.parse(execFileSync('curl', ['-s', `${url}/status.json`]).toString());
ck(lt.height === tipNow.height && lt.height > 9 && tab2.localTip().hash === tipNow.tip, `a tab opened at block ${lt.height} syncs to the coordinator's tip`);
ck(late.some(([e, d]) => e === 'fetch' && d.name === 'tip.mlf2' && d.from === 'coordinator'), 'from the heads and the model fetched whole');
const sizes = [...tab2.fs.files].filter(([k]) => k.startsWith('blocks/')).map(([, f]) => f.size);
ck(sizes.length === lt.height && sizes.every((n) => n < 4096), `holding ${sizes.length} heads and no delta`);
ck((await tab2.trainOnce()).height === lt.height, 'and it trains');
const cr = await tab2.credit(pk);
const cli = execFileSync(spel, ['credit', dir, pk]).toString();
const passed = Number(/passed +(\d+) jobs/.exec(cli)?.[1]);
ck(cr.passed > 0 && cr.passed === passed && cr.ops > 0 && cr.grade === null && cr.next === 1e15,
   `a late tab counts the first tab's credit from heads alone, as the coordinator does (${cr.passed} jobs)`);
const kf = join(dirname(dir), 'tab.key');   // beside the chain, in the test's own build directory
writeFileSync(kf, keyText(seed), { mode: 0o600 });
ck(execFileSync(spel, ['key', '--key', kf]).toString().trim() === pk && seedOf(keyText(seed)) === seed,
   'a page\'s key file is the terminal\'s key: spel key reads it to the same public key');
let bad = false;
try { seedOf('spelaeum key\nnot hex\n'); } catch { bad = true; }
ck(bad, 'a file that is not a key is refused');
// a mirror: the chain laid out by spel mirror, on a static server that knows nothing of spelaeum
const mdir = join(dirname(dir), 'mirror');
execFileSync(spel, ['mirror', dir, mdir, '--every', '100'], { stdio: ['ignore', 'ignore', 'pipe'] });
const files = createServer(async (q, res) => {
	const path = decodeURIComponent(new URL(q.url, 'http://x').pathname);
	try {
		if (path.includes('..')) throw new Error('no');
		const b = await readFile(join(mdir, path));
		res.writeHead(200, { 'Access-Control-Allow-Origin': '*' }); res.end(b);
	} catch { res.writeHead(404); res.end(); }
});
await new Promise((ok) => files.listen(0, '127.0.0.1', ok));
await m.trainOnce(); try { seal(); } catch {}   // one block more than the mirror has
const seen = [], tab3 = new Trainer({ url, files: `http://127.0.0.1:${files.address().port}/`, module, name: 'third', seed: newSeed(), on: (e, d) => seen.push([e, d]) });
const t3 = await tab3.sync(), tipEnd = JSON.parse(execFileSync('curl', ['-s', `${url}/status.json`]).toString());
const from = (n) => seen.find(([e, d]) => e === 'fetch' && d.name === n)?.[1].from;
ck(t3.height === tipEnd.height && tab3.localTip().hash === tipEnd.tip, `a tab with a mirror one block behind syncs to the tip (${t3.height})`);
ck(from('genesis.mlf2') === 'mirror' && from('ids.bin') === 'mirror' && from('tip.mlf2') === 'mirror', 'the data, genesis and the snapshot from the mirror');
ck(from(tab3.block(t3.height)) === 'coordinator' && tab3.fromUrl === 1, 'and only the block the mirror lacks from the coordinator');
files.close();
// checks: the third tab's delta in a closed round, checked by the first tab (a key with passed jobs)
await tab3.trainOnce();
execFileSync(spel, ['close', dir], { stdio: ['ignore', 'ignore', 'pipe'] });
ck((await m.tip()).phase === 'checking', 'a closed round: /tip says it is being checked');
await m.sync();
const checkedBy = await m.checkRound();
ck(checkedBy === 1, 'the first tab checks the third\'s delta in wasm and its signed verdict is taken');
// spot 1: the seal works the delta out natively anyway, and a verdict a bit off bars its key
const sealed = spawnSync(spel, ['seal', dir, '--checks', '1', '--spot', '1']).stderr.toString();
ck(/0 deltas taken from trainers' checks/.test(sealed) && !/barred/.test(sealed) && /passes|short of the difficulty/.test(sealed),
	'the seal works that delta out natively too, and the loss the tab worked out in wasm matches it to the bit');
// token positions are 64 bits in wasm32 too, where C's long is 32: a corpus of 15T tokens
// lands each window where the terminal does (ch_window, done here again in BigInt)
{
	const M = (1n << 64n) - 1n;
	const mix = (z) => { z = (z + 0x9E3779B97F4A7C15n) & M; z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & M;
		z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & M; return z ^ (z >> 31n); };
	const win = (ntok, sh, st, s, r, T) => (r % (ntok / (st * sh)) * sh + s) * st + mix(r ^ 0x5851F42D4C957F2Dn) % (st - T - 1n);
	let same = 0, n = 0;
	for (const [ntok, r] of [[15000000000000n, 12345678901234567n], [3000000000n, 0xFEDCBA9876543210n], [2000000000n, 7n]]) {
		const got = (await run(module, m.fs, ['window', `${ntok}`, '64', '4096', '5', `${r}`, '512'])).out.trim();
		same += got === `${win(ntok, 64n, 4096n, 5n, r, 512n)}`; n++;
	}
	ck(same === n, `a window past 2^31 tokens lands where the terminal's does (${same}/${n})`);
}
console.log(fails ? `test_web: ${fails} FAILED` : 'test_web: all passed');
process.exit(fails ? 1 : 0);
