// SPDX-License-Identifier: GPL-3.0-or-later — Copyright (C) 2026 Zachary Blauser
// gpu.js — the engine's gpu.h on WebGPU, for spel.wasm (web/gpu_web.c imports it
// as module "gpu"). The kernels are the engine's OpenCL ones (hako src/gpu.c,
// TSRC) in WGSL, one for one, in gpu.h's GK_ order; test-gpu.mjs holds them to
// the trainer's CPU pass. Every operand is a whole buffer and an offset in
// elements (WebGPU binds at 256-byte offsets, tensors sit anywhere), so a
// kernel's offsets ride with its numbers in a uniform: word i of the launch's
// arguments, the number of work-groups it needs in word 31. Launches queue in one
// compute pass; a read ends it, submits and waits (the C waits too: get is a JSPI
// suspending import). Without WebGPU or JSPI, openGpu gives null and gpuImports
// names no device, so the trainer stays on the CPU.

const HEAD = `
@group(0) @binding(0) var<uniform> P: array<vec4<u32>, 8>;
fn w(i: u32) -> u32 { return P[i >> 2u][i & 3u]; }
fn wi(i: u32) -> i32 { return bitcast<i32>(w(i)); }
fn wf(i: u32) -> f32 { return bitcast<f32>(w(i)); }
`;
const RED = `
var<workgroup> red: array<f32, 64>;
fn wsum(l: u32, v: f32) -> f32 {
	red[l] = v; workgroupBarrier();
	for (var h = 32u; h > 0u; h = h >> 1u) { if (l < h) { red[l] = red[l] + red[l + h]; } workgroupBarrier(); }
	let s = red[0]; workgroupBarrier();
	return s;
}
fn wmax(l: u32, v: f32) -> f32 {
	red[l] = v; workgroupBarrier();
	for (var h = 32u; h > 0u; h = h >> 1u) { if (l < h) { red[l] = max(red[l], red[l + h]); } workgroupBarrier(); }
	let s = red[0]; workgroupBarrier();
	return s;
}
`;
// a 1-D kernel's item and a row kernel's row, over a dispatch folded to fit 65535 a side
const MAIN = `@compute @workgroup_size(64) fn main(@builtin(local_invocation_id) li: vec3<u32>,
	@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>)`;
const LIN = `let grp = wg.x + wg.y * nw.x; let l = li.x; let i = grp * 64u + l;`;
const ints = (n, int) => n.map((x, k) => `@group(0) @binding(${k + 1}) var<storage, read_write> ${x}: array<${int.includes(x) ? 'i32' : 'f32'}>;`).join('\n');

const K = [
// GK_GEMM: C[M,N] (+)= op(A) op(B) over a batch of (sequence, head), 64 x 64 tiles. Unlike the
// OpenCL one, an item makes 4 neighbouring rows by 4 neighbouring columns, in four vec4s: a
// slice of A and of B is then one vec4 each a step, and nothing is an array (WGSL compilers do
// not unroll one, so a float[16] lived in slow memory: 25 GFLOPS where this does many times that)
`${ints(['A', 'B', 'C'], [])}
var<workgroup> As: array<vec4<f32>, 256>;
var<workgroup> Bs: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16, 1) fn main(@builtin(local_invocation_id) li: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
	let M = w(3); let N = w(4); let K = w(5); let sam = w(6); let sak = w(7); let sbk = w(8); let sbn = w(9);
	let ldc = w(10); let add = w(11);
	let zb = wg.z / w(12); let h = wg.z % w(12) * w(13) + w(14);
	let oa = w(0) + zb * w(15) + (h / w(17)) * w(16);
	let ob = w(1) + zb * w(18) + (h / w(20)) * w(19);
	let oc = w(2) + zb * w(21) + (h / w(23)) * w(22);
	let tx = li.x; let ty = li.y; let id = ty * 16u + tx; let at = sak != 1u; let bt = sbn != 1u;
	let m0 = wg.y * 64u; let n0 = wg.x * 64u;
	var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
	for (var k0 = 0u; k0 < K; k0 = k0 + 16u) {
		/* a vec4 an item of each slice, written whole (a component store into shared memory is a
		   read-modify-write of the vector on some compilers, so neighbours writing parts of one
		   race): four rows of A at a depth, four columns of B at a depth, the item order along
		   whichever axis is stride 1 */
		var ka = id & 15u; var rq = id >> 4u;
		if (at) { rq = id & 15u; ka = id >> 4u; }
		var va = vec4<f32>(0.0);
		let gk = k0 + ka;
		${[0, 1, 2, 3].map((j) => `{ let gm = m0 + rq * 4u + ${j}u; if (gm < M && gk < K) { va[${j}] = A[oa + gm * sam + gk * sak]; } }`).join('\n\t\t')}
		As[ka * 16u + rq] = va;
		var kb = id >> 4u; var cq = id & 15u;
		if (bt) { kb = id & 15u; cq = id >> 4u; }
		var vb = vec4<f32>(0.0);
		let gb = k0 + kb;
		${[0, 1, 2, 3].map((j) => `{ let gn = n0 + cq * 4u + ${j}u; if (gn < N && gb < K) { vb[${j}] = B[ob + gb * sbk + gn * sbn]; } }`).join('\n\t\t')}
		Bs[kb * 16u + cq] = vb;
		workgroupBarrier();
		for (var k = 0u; k < 16u; k = k + 1u) {
			let a = As[k * 16u + ty]; let b = Bs[k * 16u + tx];
			c0 = c0 + a.x * b; c1 = c1 + a.y * b;
			c2 = c2 + a.z * b; c3 = c3 + a.w * b;
		}
		workgroupBarrier();
	}
	let gn = n0 + tx * 4u;
	${[0, 1, 2, 3].map((i) => `{ let gm = m0 + ty * 4u + ${i}u;
	if (gm < M) { for (var j = 0u; j < 4u; j = j + 1u) { if (gn + j < N) { let o = oc + gm * ldc + gn + j;
		if (add != 0u) { C[o] = C[o] + c${i}[j]; } else { C[o] = c${i}[j]; } } } } }`).join('\n\t')}
}`,
// GK_EMBED
`${ints(['x', 'E', 'ids'], ['ids'])}
${MAIN} { ${LIN}
	let D = w(3); let N = w(4);
	if (i < N * D) { x[w(0) + i] = E[w(1) + u32(ids[w(2) + i / D]) * D + i % D]; }
}`,
// GK_ADD_RMS: a row a work-group: x += xa (has), out = x / rms(x) * w, r = 1 / rms
`${ints(['o', 'r', 'x', 'xa', 'g'], [])}
${RED}
${MAIN} { ${LIN}
	if (grp >= w(31)) { return; }
	let D = w(5); let eps = wf(6); let has = w(7); let xr = w(2) + grp * D;
	var ss = 0.0;
	for (var k = l; k < D; k = k + 64u) {
		var v = x[xr + k];
		if (has != 0u) { v = v + xa[w(3) + grp * D + k]; x[xr + k] = v; }
		ss = ss + v * v;
	}
	let s = 1.0 / sqrt(wsum(l, ss) / f32(D) + eps);
	if (l == 0u) { r[w(1) + grp] = s; }
	for (var k = l; k < D; k = k + 64u) { o[w(0) + grp * D + k] = x[xr + k] * s * g[w(4) + k]; }
}`,
// GK_RMS_DX
`${ints(['dx', 'dy', 'x', 'r', 'g'], [])}
${RED}
${MAIN} { ${LIN}
	if (grp >= w(31)) { return; }
	let D = w(5); let b = grp * D; let s = r[w(3) + grp];
	var d = 0.0;
	for (var k = l; k < D; k = k + 64u) { d = d + dy[w(1) + b + k] * g[w(4) + k] * x[w(2) + b + k]; }
	let c = s * s * s * wsum(l, d) / f32(D);
	for (var k = l; k < D; k = k + 64u) { dx[w(0) + b + k] = dx[w(0) + b + k] + s * dy[w(1) + b + k] * g[w(4) + k] - c * x[w(2) + b + k]; }
}`,
// GK_RMS_DW: rows in blocks of 128 into part (x: elements, y: the block)
`${ints(['part', 'dy', 'x', 'r'], [])}
${MAIN} {
	let i = wg.x * 64u + li.x; let y = wg.y; let D = w(4); let N = w(5);
	if (i >= D) { return; }
	var s = 0.0;
	for (var n = y * 128u; n < N && n < y * 128u + 128u; n = n + 1u) { s = s + dy[w(1) + n * D + i] * x[w(2) + n * D + i] * r[w(3) + n]; }
	part[w(0) + y * D + i] = s;
}`,
// GK_COLSUM
`${ints(['dw', 'part'], [])}
${MAIN} { ${LIN}
	let D = w(2);
	if (i >= D) { return; }
	var s = 0.0;
	for (var y = 0u; y < w(3); y = y + 1u) { s = s + part[w(1) + y * D + i]; }
	dw[w(0) + i] = dw[w(0) + i] + s;
}`,
// GK_ROPE
`${ints(['x', 'cs', 'sn'], [])}
${MAIN} { ${LIN}
	let N = w(3); let T = w(4); let nh = w(5); let hd = w(6); let dir = wf(7); let hh = hd / 2u;
	if (i >= N * nh * hh) { return; }
	let k = i % hh; let q = i / hh; let h = q % nh; let n = q / nh; let pos = n % T;
	let v = w(0) + (n * nh + h) * hd;
	let c = cs[w(1) + pos * hh + k]; let s = dir * sn[w(2) + pos * hh + k]; let a = x[v + k]; let b = x[v + k + hh];
	x[v + k] = a * c - b * s; x[v + k + hh] = a * s + b * c;
}`,
// GK_ATT_SM: p = softmax(S x sc) on s <= t in place, 0 after; a row a work-group
`${ints(['p'], [])}
${RED}
${MAIN} { ${LIN}
	if (grp >= w(31)) { return; }
	let T = w(1); let sc = wf(2); let t = grp % T; let b = w(0) + grp * T;
	var mx = -3.4e38;
	for (var s = l; s <= t; s = s + 64u) { mx = max(mx, p[b + s] * sc); }
	mx = wmax(l, mx);
	var sum = 0.0;
	for (var s = l; s <= t; s = s + 64u) { let e = exp(p[b + s] * sc - mx); p[b + s] = e; sum = sum + e; }
	let inv = 1.0 / wsum(l, sum);
	for (var s = l; s < T; s = s + 64u) { if (s <= t) { p[b + s] = p[b + s] * inv; } else { p[b + s] = 0.0; } }
}`,
// GK_ATT_SMB: dS = P (dP - rowsum(P dP)) x sc in place over dP
`${ints(['p', 'ds'], [])}
${RED}
${MAIN} { ${LIN}
	if (grp >= w(31)) { return; }
	let T = w(2); let sc = wf(3); let t = grp % T; let pb = w(0) + grp * T; let sb = w(1) + grp * T;
	var d = 0.0;
	for (var s = l; s <= t; s = s + 64u) { d = d + p[pb + s] * ds[sb + s]; }
	d = wsum(l, d);
	for (var s = l; s < T; s = s + 64u) { if (s <= t) { ds[sb + s] = p[pb + s] * (ds[sb + s] - d) * sc; } else { ds[sb + s] = 0.0; } }
}`,
// GK_SWIGLU_F
`${ints(['g', 'u', 'h'], [])}
${MAIN} { ${LIN}
	if (i < w(3)) { let x = g[w(0) + i]; h[w(2) + i] = x / (1.0 + exp(-x)) * u[w(1) + i]; }
}`,
// GK_SWIGLU_B
`${ints(['g', 'u', 'dA', 'dB'], [])}
${MAIN} { ${LIN}
	if (i >= w(4)) { return; }
	let x = g[w(0) + i]; let sg = 1.0 / (1.0 + exp(-x)); let dh = dA[w(2) + i];
	dB[w(3) + i] = dh * x * sg;
	dA[w(2) + i] = dh * u[w(1) + i] * sg * (1.0 + x * (1.0 - sg));
}`,
// GK_CE: a row's nll; with scale > 0 the logits become scale x dlogits
`${ints(['z', 'tgt', 'nll'], ['tgt'])}
${RED}
${MAIN} { ${LIN}
	if (grp >= w(31)) { return; }
	let V = w(3); let scale = wf(4); let y = u32(tgt[w(1) + grp]); let b = w(0) + grp * V;
	let zy = z[b + y];
	var mx = -3.4e38;
	for (var k = l; k < V; k = k + 64u) { mx = max(mx, z[b + k]); }
	mx = wmax(l, mx);
	var se = 0.0;
	for (var k = l; k < V; k = k + 64u) { let e = exp(z[b + k] - mx); se = se + e; if (scale > 0.0) { z[b + k] = e; } }
	se = wsum(l, se);
	if (l == 0u) { nll[w(2) + grp] = log(se) + mx - zy; }
	if (scale > 0.0) {
		let c = scale / se;
		for (var k = l; k < V; k = k + 64u) { var v = z[b + k] * c; if (k == y) { v = v - scale; } z[b + k] = v; }
	}
}`,
// GK_EMB_GRAD: ids holds the distinct tokens, where each one's positions start (nu + 1), the positions
`${ints(['G', 'dx', 'ids'], ['ids'])}
${MAIN} { ${LIN}
	let nu = w(3); let D = w(4);
	if (i >= nu * D) { return; }
	let u = i / D; let k = i % D; let o = w(2) + nu; let ps = o + nu + 1u;
	var s = 0.0;
	for (var j = u32(ids[o + u]); j < u32(ids[o + u + 1u]); j = j + 1u) { s = s + dx[w(1) + u32(ids[ps + j]) * D + k]; }
	let t = w(0) + u32(ids[w(2) + u]) * D + k;
	G[t] = G[t] + s;
}`,
// GK_SUMSQ: each work-group's sum of squares, striding over all the items launched
`${ints(['x', 'part'], [])}
${RED}
${MAIN} { ${LIN}
	if (grp >= w(31)) { return; }
	var s = 0.0;
	for (var j = i; j < w(2); j = j + w(31) * 64u) { let v = x[w(0) + j]; s = s + v * v; }
	s = wsum(l, s);
	if (l == 0u) { part[w(1) + grp] = s; }
}`,
// GK_ZERO
`${ints(['x'], [])}
${MAIN} { ${LIN} if (i < w(1)) { x[w(0) + i] = 0.0; } }`,
// GK_SCALE
`${ints(['x'], [])}
${MAIN} { ${LIN} if (i < w(1)) { x[w(0) + i] = x[w(0) + i] * wf(2); } }`,
// GK_ADAMW
`${ints(['p', 'g', 'mo', 've'], [])}
${MAIN} { ${LIN}
	if (i >= w(4)) { return; }
	let lr = wf(5); let decay = wf(6); let b1 = wf(7); let b2 = wf(8); let eps = wf(9); let c1 = wf(10); let c2 = wf(11);
	let gv = g[w(1) + i];
	let m = b1 * mo[w(2) + i] + (1.0 - b1) * gv; let v = b2 * ve[w(3) + i] + (1.0 - b2) * gv * gv;
	mo[w(2) + i] = m; ve[w(3) + i] = v;
	let pv = p[w(0) + i];
	p[w(0) + i] = pv - lr * ((m / c1) / (sqrt(v / c2) + eps) + decay * pv);
}`,
];
// the same product in 128 x 128 tiles, 8 x 8 an item (rows ty*4.. and 64 + ty*4.., likewise the
// columns): four vec4 loads a step feed sixteen vec4 multiply-adds, written a * b + c and not
// fma(): exact fma runs at a quarter rate on some GPUs (the 555X under Apple's driver), where the
// compiler fuses a * b + c at full rate. For products at least 128 each
// way; the 64 tiles above take the rest (attention's T x head_dim, narrow kv heads)
const R8 = [0, 1, 2, 3, 4, 5, 6, 7], C2 = [0, 1];
const GEMM128 = `${ints(['A', 'B', 'C'], [])}
var<workgroup> As: array<vec4<f32>, 512>;
var<workgroup> Bs: array<vec4<f32>, 512>;
@compute @workgroup_size(16, 16, 1) fn main(@builtin(local_invocation_id) li: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
	let M = w(3); let N = w(4); let K = w(5); let sam = w(6); let sak = w(7); let sbk = w(8); let sbn = w(9);
	let ldc = w(10); let add = w(11);
	let zb = wg.z / w(12); let h = wg.z % w(12) * w(13) + w(14);
	let oa = w(0) + zb * w(15) + (h / w(17)) * w(16);
	let ob = w(1) + zb * w(18) + (h / w(20)) * w(19);
	let oc = w(2) + zb * w(21) + (h / w(23)) * w(22);
	let tx = li.x; let ty = li.y; let id = ty * 16u + tx; let at = sak != 1u; let bt = sbn != 1u;
	let m0 = wg.y * 128u; let n0 = wg.x * 128u;
	${R8.map((r) => C2.map((j) => `var c${r}${j} = vec4<f32>(0.0);`).join(' ')).join('\n\t')}
	for (var k0 = 0u; k0 < K; k0 = k0 + 16u) {
		for (var q = 0u; q < 2u; q = q + 1u) {
			let e = id + 256u * q;
			var ka = e & 15u; var rq = e >> 4u;
			if (at) { rq = e & 31u; ka = e >> 5u; }
			var va = vec4<f32>(0.0);
			let gk = k0 + ka;
			${[0, 1, 2, 3].map((j) => `{ let gm = m0 + rq * 4u + ${j}u; if (gm < M && gk < K) { va[${j}] = A[oa + gm * sam + gk * sak]; } }`).join('\n\t\t\t')}
			As[ka * 32u + rq] = va;
			var kb = e >> 5u; var cq = e & 31u;
			if (bt) { kb = e & 15u; cq = e >> 4u; }
			var vb = vec4<f32>(0.0);
			let gb = k0 + kb;
			${[0, 1, 2, 3].map((j) => `{ let gn = n0 + cq * 4u + ${j}u; if (gn < N && gb < K) { vb[${j}] = B[ob + gb * sbk + gn * sbn]; } }`).join('\n\t\t\t')}
			Bs[kb * 32u + cq] = vb;
		}
		workgroupBarrier();
		for (var k = 0u; k < 16u; k = k + 1u) {
			let a0 = As[k * 32u + ty]; let a1 = As[k * 32u + 16u + ty];
			let b0 = Bs[k * 32u + tx]; let b1 = Bs[k * 32u + 16u + tx];
			${R8.map((r) => C2.map((j) => `c${r}${j} = c${r}${j} + a${r >> 2}[${r & 3}] * b${j};`).join(' ')).join('\n\t\t\t')}
		}
		workgroupBarrier();
	}
	${R8.map((r) => C2.map((j) => `{ let gm = m0 + ${r >> 2}u * 64u + ty * 4u + ${r & 3}u; let gn = n0 + ${j}u * 64u + tx * 4u;
	if (gm < M) { for (var u = 0u; u < 4u; u = u + 1u) { if (gn + u < N) { let o = oc + gm * ldc + gn + u;
		if (add != 0u) { C[o] = C[o] + c${r}${j}[u]; } else { C[o] = c${r}${j}[u]; } } } } }`).join('\n\t')).join('\n\t')}
}`;
const GK_GEMM = 0, SLOTS = 2048, SLOT = 256, GEMM_BIG = K.length;

// the device once a page (or test), its 18 pipelines compiled (gpu.h's 17, the big gemm); null when there is none
export async function openGpu() {
	if (typeof navigator === 'undefined' || !navigator.gpu || typeof WebAssembly.Suspending !== 'function') return null;
	const ad = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
	if (!ad) return null;
	const lim = ad.limits;
	const dev = await ad.requestDevice({ requiredLimits: {
		maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
		maxStorageBuffersPerShaderStage: Math.min(8, lim.maxStorageBuffersPerShaderStage) } });
	const g = { dev, name: (ad.info && (ad.info.description || ad.info.vendor)) || 'WebGPU', failed: null,
		pipes: [], bufs: new Map(), next: 1, enc: null, pass: null, slot: 0,
		ring: dev.createBuffer({ size: SLOTS * SLOT, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }) };
	dev.addEventListener('uncapturederror', (e) => { g.failed = g.failed || String(e.error && e.error.message || e.error); });
	dev.lost.then((i) => { g.failed = g.failed || 'device lost: ' + i.message; });
	dev.pushErrorScope('validation');
	for (const src of [...K, GEMM128]) g.pipes.push(dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: HEAD + src }), entryPoint: 'main' } }));
	const bad = await dev.popErrorScope();
	if (bad) { console.error('spel: the WebGPU kernels did not build: ' + bad.message); dev.destroy(); return null; }
	return g;
}

// end the open pass and submit what it queued
function flush(g) {
	if (!g.enc) return;
	g.pass.end();
	g.dev.queue.submit([g.enc.finish()]);
	g.enc = g.pass = null; g.slot = 0;
}

// the "gpu" module for one run of spel.wasm; mem() gives its memory. g null: no device
export function gpuImports(g, mem) {
	if (!g) return { name: () => 0, new: () => 0, del() {}, put: () => -1, get: () => -1, launch: () => -1, done() {} };
	const u8 = () => new Uint8Array(mem().buffer);
	const owned = new Set();
	const fail = (what) => { if (g && !g.failed) g.failed = what; return -1; };
	const imp = {
		name(p, cap) {
			if (!g || g.failed) return 0;
			const b = new TextEncoder().encode(g.name).subarray(0, cap);
			u8().set(b, p);
			return b.length;
		},
		new: new WebAssembly.Suspending(async (bytes) => {
			if (!g || g.failed || bytes > g.dev.limits.maxStorageBufferBindingSize) return 0;
			g.dev.pushErrorScope('out-of-memory');
			const b = g.dev.createBuffer({ size: (bytes + 3) & ~3, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
			if (await g.dev.popErrorScope()) { b.destroy(); return 0; }
			const h = g.next++;
			g.bufs.set(h, b); owned.add(h);
			return h;
		}),
		del(h) { const b = g && g.bufs.get(h); if (b) { flush(g); b.destroy(); g.bufs.delete(h); owned.delete(h); } },
		put(h, off, src, n) {
			const b = g && g.bufs.get(h);
			if (!b || g.failed || n % 4 || off % 4) return fail('put');
			flush(g);   // a write lands at once on the queue: what was queued before it must go first
			g.dev.queue.writeBuffer(b, off, mem().buffer, src, n);
			return 0;
		},
		get: new WebAssembly.Suspending(async (h, off, dst, n) => {
			const b = g && g.bufs.get(h);
			if (!b || g.failed || n % 4 || off % 4) return fail('get');
			flush(g);
			const st = g.dev.createBuffer({ size: n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
			const e = g.dev.createCommandEncoder();
			e.copyBufferToBuffer(b, off, st, 0, n);
			g.dev.queue.submit([e.finish()]);
			try { await st.mapAsync(GPUMapMode.READ); } catch (x) { st.destroy(); return fail(String(x)); }
			u8().set(new Uint8Array(st.getMappedRange()), dst);
			st.unmap(); st.destroy();
			return g.failed ? -1 : 0;
		}),
		launch(k, gx, gy, gz, n, ap) {
			if (!g || g.failed || k < 0 || k >= K.length) return fail('launch');
			const a = new Uint32Array(mem().buffer, ap, 4 * n), words = new Uint32Array(SLOT / 4), ents = [];
			for (let i = 0; i < n; i++) {
				const t = a[4 * i];
				if (t === 0) {
					const b = g.bufs.get(a[4 * i + 1]);
					if (!b) return fail('launch: no buffer');
					ents.push({ binding: ents.length + 1, resource: { buffer: b } });
				}
				words[i] = t === 0 ? a[4 * i + 2] : a[4 * i + 1];
			}
			let dx, dy = 1, dz = gz;
			let pk = k;
			if (k === GK_GEMM) {   /* the tile by the product's size: M and N are its words 3 and 4 */
				const M = words[3], N = words[4];
				if (M >= 128 && N >= 128) { pk = GEMM_BIG; dx = Math.ceil(N / 128); dy = Math.ceil(M / 128); }
				else { dx = Math.ceil(gx / 16); dy = Math.ceil(gy / 16); }
			}
			else {
				const groups = Math.ceil(gx / 64);
				if (gy > 1) { dx = groups; dy = gy; }
				else { dx = Math.min(groups, 65535); dy = Math.ceil(groups / dx); }
				words[31] = groups;
			}
			if (g.slot === SLOTS) flush(g);
			g.dev.queue.writeBuffer(g.ring, g.slot * SLOT, words);
			ents.unshift({ binding: 0, resource: { buffer: g.ring, offset: g.slot * SLOT, size: 128 } });
			g.slot++;
			if (!g.enc) { g.enc = g.dev.createCommandEncoder(); g.pass = g.enc.beginComputePass(); }
			g.pass.setPipeline(g.pipes[pk]);
			g.pass.setBindGroup(0, g.dev.createBindGroup({ layout: g.pipes[pk].getBindGroupLayout(0), entries: ents }));
			g.pass.dispatchWorkgroups(dx, dy, dz);
			return 0;
		},
	};
	// what a run left behind (an exit mid-job) goes with it
	imp.done = () => { if (!g) return; flush(g); for (const h of owned) { const b = g.bufs.get(h); if (b) b.destroy(); g.bufs.delete(h); } owned.clear(); };
	return imp;
}
