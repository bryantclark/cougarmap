// The weights page's kernel: the camera-spot score and the spot picking, ported from analyze.py so the page can
// rerun them in the browser when a weight changes. Pure functions over typed arrays (row-major, H x W); the tests
// run this file under node against the Python model (tests/test_explore.py). Keep it in step with:
//   score()  <- analyze.combine (spot, stacking ramps, habitat x season, the camera-zone Gaussian, scale/clip)
//   final()  <- analyze.apply_masks (site penalties, usable ground)
//   pick()   <- analyze.pick_candidates (6 m smoothing, peak_local_max with labels, snap, spread, zones)
//
// Dragging a slider has to feel instant, so the score is taken apart once: the spot score is linear in the five
// weights and the camera-zone Gaussian is linear, so with the stacking multiplier fixed
//   blur(sum_k w_k L_k stack hab) = sum_k w_k blur(L_k stack hab),
// five blurred layers a weight change only re-adds (Engine). The stacking bonus is a polynomial in m - 1,
// prod_f (1 + (m - 1) r_f) = sum_j (m - 1)^j e_j(r) (e_j: the elementary symmetric polynomials of the four
// ramps), so 25 blurred layers make the stacking slider instant too; they are built the first time it moves.

export const FACTORS = ["wind", "edges", "pinch", "water"]; // stacked; travel is added in but not stacked
export const WEIGHTED = [...FACTORS, "travel"];

export function ramp(a, lo, hi) {
  const t = (a - lo) / (hi - lo);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

// scipy.ndimage.gaussian_filter1d weights (truncate 4.0)
function gaussWeights(sigma) {
  const radius = Math.floor(4 * sigma + 0.5);
  const w = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    w[i + radius] = Math.exp((-0.5 * i * i) / (sigma * sigma));
    sum += w[i + radius];
  }
  for (let i = 0; i < w.length; i++) w[i] /= sum;
  return { w, radius };
}

// scipy's mode="reflect" (d c b a | a b c d | d c b a)
function reflect(i, n) {
  if (n === 1) return 0;
  const p = 2 * n;
  i = ((i % p) + p) % p;
  return i < n ? i : p - 1 - i;
}

// scipy.ndimage.gaussian_filter(a, sigma, mode="reflect"): rows (axis 0) first, then columns
export function gaussian(src, H, W, sigma) {
  const { w, radius } = gaussWeights(sigma);
  const tmp = new Float32Array(H * W);
  const out = new Float32Array(H * W);
  const idx = new Int32Array(Math.max(H, W) + 2 * radius);
  for (let i = 0; i < H + 2 * radius; i++) idx[i] = reflect(i - radius, H);
  const acc = new Float64Array(W);
  for (let r = 0; r < H; r++) {
    acc.fill(0);
    for (let k = 0; k < w.length; k++) {
      const wk = w[k], base = idx[r + k] * W;
      for (let c = 0; c < W; c++) acc[c] += wk * src[base + c];
    }
    tmp.set(acc, r * W);
  }
  for (let i = 0; i < W + 2 * radius; i++) idx[i] = reflect(i - radius, W);
  for (let r = 0; r < H; r++) {
    const row = r * W;
    for (let c = 0; c < W; c++) {
      let s = 0;
      for (let k = 0; k < w.length; k++) s += w[k] * tmp[row + idx[c + k]];
      out[row + c] = s;
    }
  }
  return out;
}

// the score's top (analyze.combine): what a cell with every factor at 1 would score before scaling
function scale(w, h) {
  const top =
    (w.wind + w.edges + w.pinch + w.water + w.travel) *
    Math.pow(w.stack_multiplier, 4) *
    (1 + h.edge_floor) *
    (1 + h.water_floor);
  return top > 0 ? (100 * h.score_scale) / top : 0;
}

function blur(a, H, W, h, res) {
  return h.zone_m > 0 ? gaussian(a, H, W, h.zone_m / res) : a;
}

// the five blurred layers blur(L_k x stack x hab) at one stacking multiplier
function basisAt(L, H, W, w, h, res) {
  const n = H * W, m = w.stack_multiplier - 1;
  const stack = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = L.habitat[i];
    for (const f of FACTORS) s *= 1 + m * ramp(L[f][i], w.stack_from, w.stack_to);
    stack[i] = s;
  }
  return WEIGHTED.map((k) => {
    const a = new Float32Array(n), x = L[k];
    for (let i = 0; i < n; i++) a[i] = x[i] * stack[i];
    return blur(a, H, W, h, res);
  });
}

// the 25 blurred layers blur(L_k x e_j(ramps) x hab), j = 0..4: poly[j][k]
function basisPoly(L, H, W, w, h, res) {
  const n = H * W;
  const e = [0, 1, 2, 3, 4].map(() => new Float32Array(n));
  const t = new Float64Array(5);
  for (let i = 0; i < n; i++) {
    t.fill(0);
    t[0] = 1;
    for (const f of FACTORS) {
      const r = ramp(L[f][i], w.stack_from, w.stack_to);
      for (let j = 4; j >= 1; j--) t[j] += r * t[j - 1];
    }
    const hab = L.habitat[i];
    for (let j = 0; j < 5; j++) e[j][i] = t[j] * hab;
  }
  return e.map((ej) =>
    WEIGHTED.map((k) => {
      const a = new Float32Array(n), x = L[k];
      for (let i = 0; i < n; i++) a[i] = x[i] * ej[i];
      return blur(a, H, W, h, res);
    }),
  );
}

// sum_t c_t B_t, scaled and clipped to 0-100
function combine(B, c, k, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let t = 0; t < B.length; t++) v += c[t] * B[t][i];
    v *= k;
    out[i] = v < 0 ? 0 : v > 100 ? 100 : v;
  }
  return out;
}

// analyze.combine's score (0-100) from the weighted layers L (wind, edges, pinch, water, travel, habitat =
// habitat x season), weights w, habitat options h and the cell size res (m)
export function score(L, H, W, w, h, res) {
  return combine(basisAt(L, H, W, w, h, res), WEIGHTED.map((k) => w[k]), scale(w, h), H * W);
}

// a penalty's strength: true/1 the model's multiplier, false/0 none, s the multiplier to the power s
function strength(on) {
  return on === true ? 1 : on === false || on == null ? 0 : on;
}

// the ranked score: score x the site penalties at their strengths, 0 off the usable ground
export function final(sc, usable, L, on) {
  const n = sc.length;
  const out = new Float32Array(n);
  const pens = ["paved", "houses", "recreation"].map((k) => [L[k], strength(on[k])]).filter(([, s]) => s !== 0);
  for (let i = 0; i < n; i++) {
    if (!usable[i]) continue;
    let v = sc[i];
    for (const [p, s] of pens) v *= s === 1 ? p[i] : Math.pow(p[i], s);
    out[i] = v;
  }
  return out;
}

// running max over [i - d, i + d] clipped to the line (van Herk / Gil-Werman: three compares a cell, any d)
function maxLine(src, n, d, out, g, hh) {
  const k = 2 * d + 1;
  const N = Math.ceil((n + 2 * d) / k) * k;
  for (let j = 0; j < N; j++) {
    const i = j - d;
    const v = i >= 0 && i < n ? src[i] : -Infinity;
    g[j] = j % k === 0 ? v : Math.max(g[j - 1], v);
  }
  for (let j = N - 1; j >= 0; j--) {
    const i = j - d;
    const v = i >= 0 && i < n ? src[i] : -Infinity;
    hh[j] = j % k === k - 1 ? v : Math.max(hh[j + 1], v);
  }
  for (let i = 0; i < n; i++) out[i] = Math.max(hh[i], g[i + 2 * d]);
}

// ndimage.maximum_filter with a (2d+1) square and mode="nearest" (separable)
export function maxFilter(a, H, W, d) {
  const out = new Float32Array(H * W);
  const len = Math.max(H, W) + 4 * d + 2;
  const g = new Float32Array(len), hh = new Float32Array(len);
  const line = new Float32Array(Math.max(H, W)), res = new Float32Array(Math.max(H, W));
  for (let r = 0; r < H; r++) {
    maxLine(a.subarray(r * W, (r + 1) * W), W, d, res, g, hh);
    out.set(res.subarray(0, W), r * W);
  }
  for (let c = 0; c < W; c++) {
    for (let r = 0; r < H; r++) line[r] = out[r * W + c];
    maxLine(line, H, d, res, g, hh);
    for (let r = 0; r < H; r++) out[r * W + c] = res[r];
  }
  return out;
}

// analyze.spread: indices to keep (best first), at most perZone within radius of each other, n in all
function spread(xy, perZone, radius, n) {
  const picked = [];
  for (let i = 0; i < xy.length; i++) {
    let near = 0;
    for (const j of picked) if (Math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) < radius) near++;
    if (near >= perZone) continue;
    picked.push(i);
    if (picked.length >= n) break;
  }
  return picked;
}

// analyze.zones: zone number (1, 2, ...) of each point
function zones(xy, radius) {
  const zone = xy.map((_, i) => i);
  for (let i = 0; i < xy.length; i++) {
    for (let j = 0; j < i; j++) {
      if (Math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) < radius) {
        zone[i] = zone[j];
        break;
      }
    }
  }
  const map = new Map();
  return zone.map((z) => {
    if (!map.has(z)) map.set(z, map.size + 1);
    return map.get(z);
  });
}

// analyze.pick_candidates on the page grid: [{row, col, score, rank, zone}]
export function pick(fin, usable, H, W, res, p) {
  const n = H * W;
  let any = false;
  for (let i = 0; i < n; i++) if (usable[i]) { any = true; break; }
  if (!any) return [];
  const sm = gaussian(fin, H, W, Math.max(0.5, 6 / res));
  const img = new Float32Array(n);
  let best = -Infinity;
  for (let i = 0; i < n; i++) {
    img[i] = usable[i] ? sm[i] : -Infinity;
    if (usable[i] && sm[i] > best) best = sm[i];
  }
  const thr = Math.min(p.peak_min, Math.max(p.peak_floor, p.peak_rel * best));
  const d = Math.max(1, Math.round(p.spacing_m / res));
  const mx = maxFilter(img, H, W, d);
  const peaks = [];
  for (let i = 0; i < n; i++) if (usable[i] && img[i] === mx[i] && img[i] > thr) peaks.push(i);
  peaks.sort((a, b) => img[b] - img[a] || a - b); // highest first, stable (row-major ties)
  // skimage's ensure_spacing: greedy, Chebyshev distance < d, at most n x 8
  const maxOut = p.n * 8;
  const kept = [];
  for (const i of peaks) {
    const r = Math.floor(i / W), c = i % W;
    let ok = true;
    for (const j of kept) {
      if (Math.max(Math.abs(Math.floor(j / W) - r), Math.abs((j % W) - c)) < d) { ok = false; break; }
    }
    if (ok) kept.push(i);
    if (kept.length >= maxOut) break;
  }
  // snap to the best cell within ~10 m
  const k = Math.max(1, Math.floor(10 / res));
  let cands = kept.map((i) => {
    const pr = Math.floor(i / W), pc = i % W;
    let br = pr, bc = pc, bv = -Infinity;
    for (let r = Math.max(pr - k, 0); r <= Math.min(pr + k, H - 1); r++) {
      for (let c = Math.max(pc - k, 0); c <= Math.min(pc + k, W - 1); c++) {
        if (fin[r * W + c] > bv) { bv = fin[r * W + c]; br = r; bc = c; }
      }
    }
    return { row: br, col: bc, score: bv };
  });
  cands = cands.map((c, i) => [c, i]).sort((a, b) => b[0].score - a[0].score || a[1] - b[1]).map((x) => x[0]);
  const xy = cands.map((c) => [(c.col + 0.5) * res, -(c.row + 0.5) * res]);
  const keep = spread(xy, p.per_zone, p.zone_radius_m, p.n);
  const z = zones(keep.map((i) => xy[i]), p.zone_radius_m);
  return keep.map((i, j) => ({ ...cands[i], rank: j + 1, zone: z[j] }));
}

// One area, many slider settings: the blurred layers are built once, each run only re-adds them.
// run(w, on, n, land) -> {score, final, spots, finalPrivate, privateSpots}; land "public", "private" or "both"
// (the private list needs L.usable_private)
export function engine(L, data) {
  const { height: H, width: W, res } = data.grid;
  const h = data.habitat, n = H * W;
  let at = null, atM = NaN, poly = null;
  function scoreFor(w) {
    const k = scale(w, h);
    if (w.stack_multiplier === atM) return combine(at, WEIGHTED.map((f) => w[f]), k, n);
    if (at === null) {
      at = basisAt(L, H, W, w, h, res);
      atM = w.stack_multiplier;
      return combine(at, WEIGHTED.map((f) => w[f]), k, n);
    }
    if (poly === null) poly = basisPoly(L, H, W, w, h, res).flat();
    const m = w.stack_multiplier - 1, c = [];
    for (let j = 0; j < 5; j++) for (const f of WEIGHTED) c.push(w[f] * Math.pow(m, j));
    return combine(poly, c, k, n);
  }
  function run(w, on, count, land = "both") {
    const sc = scoreFor(w);
    const p = { ...data.pick, n: count };
    const out = { score: sc, spots: [] };
    if (land !== "private") {
      out.final = final(sc, L.usable, L, on);
      out.spots = pick(out.final, L.usable, H, W, res, p);
    }
    if (L.usable_private && land !== "public") {
      out.finalPrivate = final(sc, L.usable_private, L, on);
      out.privateSpots = pick(out.finalPrivate, L.usable_private, H, W, res, p);
    }
    return out;
  }
  return { run };
}

// everything for one set of slider values: weights w, penalty strengths on {paved, houses, recreation} (true/false
// or 0-2), spot count n
export function run(L, data, w, on, n) {
  return engine(L, data).run(w, on, n);
}

// page cell centre -> [lat, lon] (explore.georef's forward fit)
export function cellLatLon(geo, H, W, row, col) {
  const u = col / W - 0.5, v = row / H - 0.5;
  const t = [1, u, v, u * u, u * v, v * v];
  let lon = 0, lat = 0;
  for (let i = 0; i < 6; i++) { lon += geo.fwd[0][i] * t[i]; lat += geo.fwd[1][i] * t[i]; }
  return [lat, lon];
}

// lat/lon -> fractional page [row, col] (the inverse fit)
export function latLonCell(geo, lat, lon) {
  const [west, south, east, north] = geo.bounds;
  const u = (lon - west) / (east - west) - 0.5, v = (north - lat) / (north - south) - 0.5;
  const t = [1, u, v, u * u, u * v, v * v];
  let c = 0, r = 0;
  for (let i = 0; i < 6; i++) { c += geo.inv[0][i] * t[i]; r += geo.inv[1][i] * t[i]; }
  return [r, c];
}
