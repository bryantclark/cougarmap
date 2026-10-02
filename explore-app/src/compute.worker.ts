// The scoring thread: decodes the area's layers once, then reruns the kernel for each slider setting and paints the
// score heat, so the map and the sliders never wait on it.
import { cellLatLon, engine, final, latLonCell, type KSpot, type Layers } from "./kernel";
import { FACTOR_KEYS, PENALTY_KEYS, type Payload, type Settings, type Spot } from "./types";

type In =
  | { type: "init"; data: Payload }
  | ({ type: "run"; id: number; heat: boolean } & Settings)
  | { type: "factor"; key: string; rgb: [number, number, number] }
  | { type: "land"; public: boolean; private: boolean; reach: boolean };

let data: Payload;
let L: Layers;
let E: ReturnType<typeof engine>;
let heatIndex: Int32Array;
let ones: Uint8Array; // heat pixel (lon/lat aligned) -> page cell, or -1
let H = 0, W = 0;

async function decode(layer: Payload["layers"][string]): Promise<Float32Array | Uint8Array> {
  const bin = atob(layer.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  const buf = await new Response(stream).arrayBuffer();
  const q = layer.dtype === "uint16" ? new Uint16Array(buf) : new Uint8Array(buf);
  if (layer.dtype === "uint8" && layer.scale === 1) return q as Uint8Array;
  const out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) out[i] = q[i] * layer.scale;
  return out;
}

function buildHeatIndex() {
  const [west, south, east, north] = data.geo.bounds;
  heatIndex = new Int32Array(W * H).fill(-1);
  for (let j = 0; j < H; j++) {
    const lat = north - ((j + 0.5) / H) * (north - south);
    for (let i = 0; i < W; i++) {
      const lon = west + ((i + 0.5) / W) * (east - west);
      const [r, c] = latLonCell(data.geo, lat, lon);
      const rr = Math.round(r), cc = Math.round(c);
      if (rr >= 0 && rr < H && cc >= 0 && cc < W) heatIndex[j * W + i] = rr * W + cc;
    }
  }
}

// the heat ramp: deep violet -> ember -> pale gold, fading in from transparent (magma-like, reads over imagery)
const STOPS: [number, number, number, number][] = [
  [0.0, 120, 28, 96],
  [0.35, 196, 48, 82],
  [0.6, 238, 104, 52],
  [0.82, 250, 176, 64],
  [1.0, 255, 240, 170],
];
const LUT = new Uint8ClampedArray(256 * 4);
for (let i = 0; i < 256; i++) {
  const v = i / 255;
  let k = 1;
  while (k < STOPS.length - 1 && v > STOPS[k][0]) k++;
  const a = STOPS[k - 1], b = STOPS[k], t = Math.min(1, Math.max(0, (v - a[0]) / (b[0] - a[0])));
  for (let c = 0; c < 3; c++) LUT[4 * i + c] = a[c + 1] + t * (b[c + 1] - a[c + 1]);
  LUT[4 * i + 3] = v < 0.16 ? 0 : Math.round(240 * Math.pow((v - 0.16) / 0.84, 0.75));
}

// the score heat over the whole area (site penalties applied): full strength on the shown land's usable ground
// (public, or private), faded elsewhere, scaled to the 99.5th percentile so one hot cell can't
// wash the rest out
async function heat(sc: Float32Array, on: Settings["pens"], priv: boolean): Promise<ImageBitmap> {
  const v = final(sc, ones, L, on);
  const sample: number[] = [];
  for (let i = 0; i < v.length; i += 7) if (v[i] > 0) sample.push(v[i]);
  sample.sort((a, b) => a - b);
  const top = sample.length ? sample[Math.min(sample.length - 1, Math.floor(sample.length * 0.995))] : 0;
  const img = new ImageData(W, H);
  const px = img.data;
  if (top > 0) {
    const pub = L.usable, pv = L.usable_private;
    for (let p = 0; p < W * H; p++) {
      const i = heatIndex[p];
      if (i < 0 || v[i] <= 0) continue;
      const q = Math.min(255, Math.round((v[i] / top) * 255)) * 4;
      const open = priv ? pv && pv[i] : pub[i];
      px[4 * p] = LUT[q];
      px[4 * p + 1] = LUT[q + 1];
      px[4 * p + 2] = LUT[q + 2];
      px[4 * p + 3] = open ? LUT[q + 3] : LUT[q + 3] * 0.55;
    }
  }
  return createImageBitmap(img);
}

// one layer alone: transparent where low, its colour (lightening toward the top) where high, scaled to its 99th
// percentile inside the area
async function factorMap(key: string, rgb: [number, number, number]): Promise<ImageBitmap> {
  const a = L[key];
  const sample: number[] = [];
  for (let i = 0; i < a.length; i += 5) if (a[i] > 0) sample.push(a[i]);
  sample.sort((x, y) => x - y);
  const top = sample.length ? sample[Math.floor(sample.length * 0.99)] || sample[sample.length - 1] : 0;
  const img = new ImageData(W, H);
  const px = img.data;
  if (top > 0) {
    for (let p = 0; p < W * H; p++) {
      const i = heatIndex[p];
      if (i < 0) continue;
      const v = Math.min(1, a[i] / top);
      if (v < 0.06) continue;
      const lift = Math.max(0, v - 0.6) / 0.4; // the top 40% fades toward white
      px[4 * p] = rgb[0] + (255 - rgb[0]) * lift * 0.55;
      px[4 * p + 1] = rgb[1] + (255 - rgb[1]) * lift * 0.55;
      px[4 * p + 2] = rgb[2] + (255 - rgb[2]) * lift * 0.55;
      px[4 * p + 3] = Math.round(235 * Math.pow((v - 0.06) / 0.94, 0.7));
    }
  }
  return createImageBitmap(img);
}

// land tints: public green, private amber, the ground within the walk limit blue
async function landMap(m: { public: boolean; private: boolean; reach: boolean }): Promise<ImageBitmap> {
  const img = new ImageData(W, H);
  const px = img.data;
  const tints: [Float32Array | Uint8Array | undefined, number, number, number, number][] = [
    [m.public ? L.public : undefined, 86, 196, 110, 0.34],
    [m.private ? L.private : undefined, 236, 160, 70, 0.3],
    [m.reach ? L.reach : undefined, 90, 160, 250, 0.3],
  ];
  for (let p = 0; p < W * H; p++) {
    const i = heatIndex[p];
    if (i < 0) continue;
    let r = 0, g = 0, b = 0, al = 0;
    for (const [a, tr, tg, tb, ta] of tints) {
      if (!a || a[i] <= 0) continue;
      const k = ta * Math.min(1, a[i]);
      // "over" compositing of the tints
      r = tr * k + r * (1 - k);
      g = tg * k + g * (1 - k);
      b = tb * k + b * (1 - k);
      al = k + al * (1 - k);
    }
    if (al <= 0) continue;
    px[4 * p] = r / al;
    px[4 * p + 1] = g / al;
    px[4 * p + 2] = b / al;
    px[4 * p + 3] = Math.round(al * 255);
  }
  return createImageBitmap(img);
}

const hab = () => (1 + data.habitat.edge_floor) * (1 + data.habitat.water_floor);

function describe(spots: KSpot[]): Spot[] {
  return spots.map((s) => {
    const i = s.row * W + s.col;
    const [lat, lon] = cellLatLon(data.geo, H, W, s.row, s.col);
    return {
      rank: s.rank,
      zone: s.zone,
      score: s.score,
      lat,
      lon,
      factors: Object.fromEntries(FACTOR_KEYS.map((k) => [k, L[k][i]])) as Spot["factors"],
      habitat: L.habitat[i] / hab(),
      penalties: Object.fromEntries(PENALTY_KEYS.map((k) => [k, L[k][i]])) as Spot["penalties"],
    };
  });
}

self.onmessage = async (e: MessageEvent<In>) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      data = m.data;
      ({ height: H, width: W } = data.grid);
      L = {};
      for (const [k, v] of Object.entries(data.layers)) L[k] = await decode(v);
      E = engine(L, data);
      ones = new Uint8Array(H * W).fill(1);
      buildHeatIndex();
      self.postMessage({ type: "ready" });
      return;
    }
    if (m.type === "factor" || m.type === "land") {
      if (m.type === "factor" && !L[m.key]) return;
      const bitmap = m.type === "factor" ? await factorMap(m.key, m.rgb) : await landMap(m);
      (self as unknown as Worker).postMessage({ type: m.type, key: m.type === "factor" ? m.key : "land", bitmap }, [bitmap]);
      return;
    }
    const t0 = performance.now();
    const w = { ...data.weights, ...m.shares, stack_multiplier: m.stack };
    const r = E.run(w, m.pens, m.n, m.priv ? "private" : "public");
    const spots = describe(m.priv ? (r.privateSpots ?? []) : r.spots); // the shown land's spots
    const ms = performance.now() - t0;
    const bitmap = m.heat ? await heat(r.score, m.pens, m.priv) : undefined;
    const msg = { type: "result", id: m.id, spots, ms, heat: bitmap };
    (self as unknown as Worker).postMessage(msg, bitmap ? [bitmap] : []);
  } catch (err) {
    self.postMessage({ type: "error", message: String((err as Error)?.message ?? err) });
  }
};
