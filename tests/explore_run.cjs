// Runs the interactive page's kernel (src/cougarmap/explore.js) under node, for tests/test_explore.py.
//   node explore_run.cjs <kernel.js> <payload.json> [settings.json]
// settings: {weights: {...}, on: {paved, houses, recreation}, n, final: true}; missing keys take the model's.
// Prints {spots: [{row, col, score, rank, zone, lat, lon}], final?: [...] (the ranked score, row-major)}.
"use strict";
const fs = require("fs");
const zlib = require("zlib");

const K = require(process.argv[2]);
const data = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const settings = process.argv[4] ? JSON.parse(fs.readFileSync(process.argv[4], "utf8")) : {};

function decode(layer) {
  const raw = zlib.inflateSync(Buffer.from(layer.data, "base64"));
  const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  const q = layer.dtype === "uint16" ? new Uint16Array(ab) : new Uint8Array(ab);
  if (layer.dtype === "uint8" && layer.scale === 1) return q;
  const out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) out[i] = q[i] * layer.scale;
  return out;
}

const L = {};
for (const [k, v] of Object.entries(data.layers)) L[k] = decode(v);
const w = { ...data.weights, ...(settings.weights || {}) };
const on = { paved: true, houses: true, recreation: true, ...(settings.on || {}) };
const n = settings.n || data.pick.n;
const t0 = process.hrtime.bigint();
const r = K.run(L, data, w, on, n);
const ms = Number(process.hrtime.bigint() - t0) / 1e6;
const { height: H, width: W } = data.grid;
const spots = r.spots.map((s) => {
  const [lat, lon] = K.cellLatLon(data.geo, H, W, s.row, s.col);
  return { ...s, lat, lon };
});
const out = { spots, ms };
if (settings.final) out.final = Array.from(r.final);
process.stdout.write(JSON.stringify(out));
