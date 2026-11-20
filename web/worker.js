// SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser
// worker.js — the page's trainer off the main thread: a job on the CPU is one wasm call
// that runs for minutes, and on the page it would freeze everything until it ended.
// In: { go: { url, name, seed, gpu } }, once. Out: { event, data }, the Trainer's events
// (trainer.js) as they come, plus device (the GPU's name or null), key (this key, public)
// and credit (this key's, after each round it sent). The page stops it by terminating it.
import { Trainer } from './trainer.js';

let t = null;
const post = (event, data) => postMessage({ event, data });
async function credit() { try { post('credit', await t.credit(await t.publicKey())); } catch {} }

onmessage = async (e) => {
	const go = e.data && e.data.go;
	if (!go || t) return;
	try {
		const module = await WebAssembly.compileStreaming(fetch('spel.wasm'));
		t = new Trainer({ url: go.url, name: go.name, seed: go.seed, gpu: go.gpu, module,
			on: (event, data) => { post(event, data); if (event === 'wait') credit(); } });
		post('device', await t.device());
		post('key', await t.publicKey());
		await t.loop();
	} catch (err) { post('error', String(err && err.message || err)); }
};
