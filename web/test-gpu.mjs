// SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser
// test-gpu.mjs — the page's GPU training (web/gpu.js, web/gpu_web.c) held to the trainer's
// CPU pass inside spel.wasm itself (its gpucheck command), then a job's tokens a second on
// each. Needs WebGPU and JSPI: deno run --unstable-webgpu --allow-read web/test-gpu.mjs.
// Skips, passing, where there is no GPU.
import { Fs, run } from './wasi.js';
import { openGpu } from './gpu.js';

const module = await WebAssembly.compile(await Deno.readFile(new URL('./spel.wasm', import.meta.url)));
const gpu = await openGpu();
if (!gpu) { console.log('test-gpu: no WebGPU (or no JSPI) here, skipped'); Deno.exit(0); }
const r = await run(module, new Fs(), ['gpucheck'], { HAKM_THREADS: '1' }, gpu);
console.log(r.out.trimEnd());
if (r.err) console.log(r.err.trimEnd());
const none = await run(module, new Fs(), ['gpu']);
const ok = r.code === 0 && none.out.trim() === 'cpu';
console.log(ok ? 'test-gpu: the page trains on the GPU as the CPU does' : `test-gpu: FAILED (exit ${r.code}, without a device: ${none.out.trim()})`);
Deno.exit(ok ? 0 : 1);
